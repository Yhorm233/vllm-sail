// SPDX-License-Identifier: Apache-2.0
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <hggc.h>
#include <hggc_runtime_api.h>

#include <cstdint>
#include <limits>
#include <new>
#include <unordered_map>

namespace {
struct Segment {
  HGdeviceptr address = 0;
  size_t size = 0;
  HGdevice device = 0;
  HGmemGenericAllocationHandle handle{};
  bool has_handle = false;
  bool mapped = false;
  bool accessible = false;
};

// The GIL protects the registry and callbacks, including in PyTorch's C ABI
// entry points. Records survive pool trimming until PyTorch frees the VA.
std::unordered_map<HGdeviceptr, Segment> segments;
PyObject* on_allocate = nullptr;
PyObject* on_free = nullptr;

bool check(HGresult result, const char* operation) {
  if (result == HGGC_SUCCESS) {
    return true;
  }
  const char* message = nullptr;
  hgGetErrorString(result, &message);
  PyErr_Format(PyExc_RuntimeError, "%s failed (%d): %s", operation,
               static_cast<int>(result), message ? message : "unknown error");
  return false;
}

bool check_runtime(hggcError_t result, const char* operation) {
  if (result == hggcSuccess) {
    return true;
  }
  PyErr_Format(PyExc_RuntimeError, "%s failed: %s", operation,
               hggcGetErrorString(result));
  return false;
}

class DeviceContext {
 public:
  explicit DeviceContext(HGdevice device) : device_(device) {
    HGcontext current = nullptr;
    if (!check(hgCtxGetCurrent(&current), "hgCtxGetCurrent")) {
      return;
    }
    if (current) {
      HGdevice active;
      if (!check(hgCtxGetDevice(&active), "hgCtxGetDevice")) {
        return;
      }
      if (active == device) {
        ready_ = true;
        return;
      }
    }
    retained_ = check(hgDevicePrimaryCtxRetain(&context_, device),
                      "hgDevicePrimaryCtxRetain");
    pushed_ = retained_ && check(hgCtxPushCurrent(context_), "hgCtxPushCurrent");
    ready_ = pushed_;
  }

  ~DeviceContext() {
    if (pushed_) {
      HGcontext popped;
      hgCtxPopCurrent(&popped);
    }
    if (retained_) {
      hgDevicePrimaryCtxRelease(device_);
    }
  }

  explicit operator bool() const { return ready_; }

 private:
  HGdevice device_;
  HGcontext context_ = nullptr;
  bool ready_ = false;
  bool retained_ = false;
  bool pushed_ = false;
};

HGmemAllocationProp properties(HGdevice device) {
  HGmemAllocationProp prop{};
  prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  return prop;
}

bool unmap(Segment& segment) {
  if (segment.mapped) {
    if (!check(hgMemUnmap(segment.address, segment.size), "hgMemUnmap")) {
      return false;
    }
    segment.mapped = false;
    segment.accessible = false;
  }
  if (segment.has_handle) {
    if (!check(hgMemRelease(segment.handle), "hgMemRelease")) {
      return false;
    }
    segment.has_handle = false;
  }
  return true;
}

bool free_segment(Segment& segment) {
  return unmap(segment) &&
         check(hgMemAddressFree(segment.address, segment.size),
               "hgMemAddressFree");
}

template <typename Cleanup>
void preserve_error(Cleanup cleanup) {
  PyObject* type;
  PyObject* value;
  PyObject* traceback;
  PyErr_Fetch(&type, &value, &traceback);
  if (!cleanup()) {
    PyErr_WriteUnraisable(Py_None);
  }
  PyErr_Restore(type, value, traceback);
}

bool map(Segment& segment) {
  auto prop = properties(segment.device);
  if (!check(hgMemCreate(&segment.handle, segment.size, &prop, 0),
             "hgMemCreate")) {
    return false;
  }
  segment.has_handle = true;
  if (!check(hgMemMap(segment.address, segment.size, 0, segment.handle, 0),
             "hgMemMap")) {
    preserve_error([&] { return unmap(segment); });
    return false;
  }
  segment.mapped = true;
  HGmemAccessDesc access{};
  access.location = prop.location;
  access.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (!check(hgMemSetAccess(segment.address, segment.size, &access, 1),
             "hgMemSetAccess")) {
    preserve_error([&] { return unmap(segment); });
    return false;
  }
  segment.accessible = true;
  return true;
}

void* allocate(size_t bytes, int device) {
  DeviceContext context(device);
  if (!context) {
    return nullptr;
  }
  auto prop = properties(device);
  size_t granularity = 0;
  if (!check(hgMemGetAllocationGranularity(
                 &granularity, &prop, HG_MEM_ALLOC_GRANULARITY_MINIMUM),
             "hgMemGetAllocationGranularity")) {
    return nullptr;
  }
  if (!bytes || !granularity ||
      bytes > std::numeric_limits<size_t>::max() - (granularity - 1)) {
    PyErr_SetString(PyExc_ValueError, "invalid HGGC allocation size");
    return nullptr;
  }
  Segment segment{};
  segment.device = device;
  segment.size = ((bytes + granularity - 1) / granularity) * granularity;
  if (!check(hgMemAddressReserve(&segment.address, segment.size, 0, 0, 0),
             "hgMemAddressReserve")) {
    return nullptr;
  }
  if (!map(segment)) {
    preserve_error([&] { return free_segment(segment); });
    return nullptr;
  }
  try {
    segments.emplace(segment.address, segment);
  } catch (const std::bad_alloc&) {
    PyErr_NoMemory();
    preserve_error([&] { return free_segment(segment); });
    return nullptr;
  }
  // Upstream's handle tuple uses the last field as an opaque integer.
  PyObject* result = PyObject_CallFunction(
      on_allocate, "((iKKK))", device,
      static_cast<unsigned long long>(segment.size),
      static_cast<unsigned long long>(segment.address),
      static_cast<unsigned long long>(segment.address));
  if (!result) {
    preserve_error([&] { return free_segment(segment); });
    segments.erase(segment.address);
    return nullptr;
  }
  Py_DECREF(result);
  return reinterpret_cast<void*>(static_cast<uintptr_t>(segment.address));
}

Segment* lookup(unsigned long long address) {
  auto it = segments.find(address);
  if (it == segments.end()) {
    PyErr_Format(PyExc_KeyError, "unknown HGGC allocation %llu", address);
    return nullptr;
  }
  return &it->second;
}

PyObject* init_module(PyObject*, PyObject* args) {
  PyObject* allocate_callback;
  PyObject* free_callback;
  if (!PyArg_ParseTuple(args, "OO", &allocate_callback, &free_callback)) {
    return nullptr;
  }
  if (!PyCallable_Check(allocate_callback) || !PyCallable_Check(free_callback)) {
    return PyErr_Format(PyExc_TypeError, "allocator callbacks must be callable");
  }
  if (on_allocate == allocate_callback && on_free == free_callback) {
    Py_RETURN_NONE;
  }
  if (!segments.empty()) {
    return PyErr_Format(PyExc_RuntimeError, "HGGC allocator already has live segments");
  }
  Py_XSETREF(on_allocate, Py_NewRef(allocate_callback));
  Py_XSETREF(on_free, Py_NewRef(free_callback));
  Py_RETURN_NONE;
}

PyObject* suspend(PyObject*, PyObject* args) {
  unsigned long long address, backup;
  if (!PyArg_ParseTuple(args, "KK", &address, &backup)) {
    return nullptr;
  }
  auto* segment = lookup(address);
  if (!segment) {
    return nullptr;
  }
  DeviceContext context(segment->device);
  if (!context) {
    return nullptr;
  }
  if (segment->mapped) {
    if (!check_runtime(hggcDeviceSynchronize(), "hggcDeviceSynchronize")) {
      return nullptr;
    }
    if (backup &&
        !check_runtime(hggcMemcpy(reinterpret_cast<void*>(backup),
                                  reinterpret_cast<void*>(address),
                                  segment->size, hggcMemcpyDeviceToHost),
                       "hggcMemcpyDeviceToHost")) {
      return nullptr;
    }
  }
  if (!unmap(*segment)) {
    return nullptr;
  }
  Py_RETURN_NONE;
}

PyObject* resume(PyObject*, PyObject* args) {
  unsigned long long address, backup;
  if (!PyArg_ParseTuple(args, "KK", &address, &backup)) {
    return nullptr;
  }
  auto* segment = lookup(address);
  if (!segment) {
    return nullptr;
  }
  DeviceContext context(segment->device);
  if (!context) {
    return nullptr;
  }
  if (!segment->accessible && (!unmap(*segment) || !map(*segment))) {
    return nullptr;
  }
  if (backup &&
      !check_runtime(hggcMemcpy(reinterpret_cast<void*>(address),
                                reinterpret_cast<void*>(backup),
                                segment->size, hggcMemcpyHostToDevice),
                     "hggcMemcpyHostToDevice")) {
    preserve_error([&] { return unmap(*segment); });
    return nullptr;
  }
  Py_RETURN_NONE;
}

PyObject* is_mapped(PyObject*, PyObject* args) {
  unsigned long long address;
  if (!PyArg_ParseTuple(args, "K", &address)) {
    return nullptr;
  }
  auto* segment = lookup(address);
  return segment ? PyBool_FromLong(segment->mapped) : nullptr;
}

PyMethodDef methods[] = {
    {"init_module", init_module, METH_VARARGS, nullptr},
    {"suspend", suspend, METH_VARARGS, nullptr},
    {"resume", resume, METH_VARARGS, nullptr},
    {"is_mapped", is_mapped, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_cumem_allocator", nullptr, -1, methods,
    nullptr, nullptr, nullptr, nullptr,
};
}  // namespace

extern "C" __attribute__((visibility("default"))) void* my_malloc(
    size_t bytes, int device, hggcStream_t) {
  auto gil = PyGILState_Ensure();
  void* result = nullptr;
  if (on_allocate) {
    result = allocate(bytes, device);
  } else {
    PyErr_SetString(PyExc_RuntimeError, "HGGC allocator callbacks are missing");
  }
  if (PyErr_Occurred()) {
    PyErr_WriteUnraisable(Py_None);
  }
  PyGILState_Release(gil);
  return result;
}

extern "C" __attribute__((visibility("default"))) void my_free(
    void* address, size_t, int, hggcStream_t) {
  auto gil = PyGILState_Ensure();
  auto it = segments.find(reinterpret_cast<uintptr_t>(address));
  if (it != segments.end()) {
    auto& segment = it->second;
    DeviceContext context(segment.device);
    if (context &&
        (!segment.mapped ||
         check_runtime(hggcDeviceSynchronize(), "hggcDeviceSynchronize")) &&
        free_segment(segment)) {
      auto ptr = segment.address;
      segments.erase(it);
      PyObject* result = PyObject_CallFunction(on_free, "K",
                                              static_cast<unsigned long long>(ptr));
      Py_XDECREF(result);
    }
  }
  if (PyErr_Occurred()) {
    PyErr_WriteUnraisable(Py_None);
  }
  PyGILState_Release(gil);
}

PyMODINIT_FUNC PyInit__cumem_allocator() { return PyModule_Create(&module); }
