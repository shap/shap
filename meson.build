project('shap', 'c', 'cpp', 'cuda',
  version: '0.45.0',
  default_options: ['warning_level=2', 'cpp_std=c++14']
)

# Dependencies
py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()
numpy_dep = dependency('numpy')

# Config options
with_cuda = get_option('with_cuda')
with_binary = get_option('with_binary')

# Regular C++ extension
if with_binary
  py.extension_module('_cext',
    'shap/cext/_cext.cc',
    dependencies: [py_dep, numpy_dep],
    install: true,
    subdir: 'shap'
  )
endif

# CUDA extension
if with_cuda
  # Try to find CUDA
  cuda_dep = dependency('cuda', required: false)
  
  if cuda_dep.found()
    cuda = import('unstable-cuda')
    nvcc = find_program('nvcc', required: false)
    
    if nvcc.found()
      # CUDA compiler flags
      cuda_args = [
        '--std=c++14',
        '--expt-extended-lambda',
        '--expt-relaxed-constexpr',
        '-gencode=arch=compute_60,code=sm_60',
        '-gencode=arch=compute_70,code=sm_70',
        '-gencode=arch=compute_75,code=sm_75',
        '-gencode=arch=compute_75,code=compute_75',
        '-gencode=arch=compute_80,code=sm_80'
      ]
      
      # Platform-specific options
      if host_machine.system() == 'windows'
        cuda_lib_path = join_paths(cuda_dep.get_variable(pkgconfig: 'prefix'), 'lib', 'x64')
      else
        cuda_lib_path = join_paths(cuda_dep.get_variable(pkgconfig: 'prefix'), 'lib64')
        cuda_args += ['-Xcompiler', '-fPIC']
      endif
      
      # Compile the CUDA library
      gpu_lib = custom_target('gpu_treeshap',
        input: 'shap/cext/_cext_gpu.cu',
        output: 'lib_cext_gpu.a',
        command: [
          nvcc, '@INPUT@', '-c', '-o', '@OUTPUT@',
          '--include-path', py.get_path('include'),
        ] + cuda_args
      )
      
      # Create the Python extension that links against the CUDA library
      py.extension_module('_cext_gpu',
        'shap/cext/_cext_gpu.cc',
        dependencies: [py_dep, numpy_dep, cuda_dep],
        link_with: gpu_lib,
        link_args: ['-lcudart'],
        link_depends: [gpu_lib],
        install: true,
        subdir: 'shap'
      )
    endif
  endif
endif

# Kernel library (Cython extension)
cython = find_program('cython3', 'cython', required: false)
if cython.found()
  kernel_lib_pyx = custom_target('_kernel_lib.c',
    input: 'shap/explainers/_kernel_lib.pyx',
    output: '_kernel_lib.c',
    command: [cython, '@INPUT@', '-o', '@OUTPUT@']
  )

  py.extension_module('_kernel_lib',
    kernel_lib_pyx,
    dependencies: [py_dep, numpy_dep],
    install: true,
    subdir: 'shap'
  )
endif