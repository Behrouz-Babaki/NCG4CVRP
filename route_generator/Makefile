.PHONY: clean

route_generator.cpython-38-x86_64-linux-gnu.so: src/rg_wrapper.pyx src/rgen.cpp src/rgen.hpp
	python setup.py build_ext --inplace

clean:
	@ rm route_generator.*.so \
	@ rm src/rg_wrapper.cpp       \
	@ rm -rf build
