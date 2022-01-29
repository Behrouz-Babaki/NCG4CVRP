.PHONY: clean

esprc.cpython-38-x86_64-linux-gnu.so: src/esprc_wrapper.pyx src/esprc.cpp src/esprc.hpp
	python setup.py build_ext --inplace

clean:
	@ rm esprc.*.so            \
	@ rm src/esprc_wrapper.cpp \
	@ rm -rf build
