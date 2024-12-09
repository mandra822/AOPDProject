cmake_build: conan_install
	cmake -S . -B ./build -DCMAKE_TOOLCHAIN_FILE="build/conan_toolchain.cmake" -DENABLE_TESTING=TRUE -DCMAKE_BUILD_TYPE=Debug  -DCMAKE_CUDA_ARCHITECTURES=75 
	cmake --build ./build 
conan_install:
	conan install . --build=missing  -of build -s build_type=Debug
test: cmake_build
	./build/src/main/tests/main_module_test
clean:
	rm -rf ./build/*


