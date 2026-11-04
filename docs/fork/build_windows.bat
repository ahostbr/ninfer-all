@echo off
rem Offline build of ninfer-all for sm_120a with CUDA 13.0.2. Opens no connection of its own
rem (vcpkg is not run: the prebuilt FFmpeg tree is named in NINFER_VCPKG_TREE).
setlocal
set "PROCESSOR_ARCHITECTURE=AMD64"
set "VSCMD_SKIP_SENDTELEMETRY=1"
set "ROOT=G:\REPO_CLONES\ninfer-all"
set "BUILD=C:\Users\Ryan\ninfer-all-build144"
set "LOGDIR=G:\REPO_CLONES\ninfer-all-work"
set "CUDA_HOME=C:\Strata-data\toolchain\cuda-13.0.2"
set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44 >nul || exit /b 1
rem put CUDA 13.0.2 first so nothing resolves the machine's 13.2
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "CUDA_PATH=%CUDA_HOME%"
set "CUDACXX=%CUDA_HOME%\bin\nvcc.exe"
"%CMAKE%" -G Ninja "-DCMAKE_MAKE_PROGRAM=%NINJA%" -S "%ROOT%" -B "%BUILD%" -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=120a -DBUILD_TESTING=ON -DNINFER_ALLOW_CUDA_13_0=ON ^
  "-DCUDAToolkit_ROOT=%CUDA_HOME%" "-DCMAKE_CUDA_COMPILER=%CUDA_HOME%\bin\nvcc.exe" ^
  "-DNINFER_VCPKG_TREE=C:/vcpkg-work/installed/x64-windows" > "%LOGDIR%\configure.log" 2>&1
echo CONFIGURE_EXIT=%ERRORLEVEL%
if errorlevel 1 exit /b 1
"%CMAKE%" --build "%BUILD%" -j 12 -- -k 0 > "%LOGDIR%\build.log" 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
