.PHONY: build run
default: build
CONFIG=Release
build:
	cmake -B./build -G "Ninja Multi-Config"
	cmake --build ./build --config $(CONFIG) --target all --
run: build
	./build/bin/$(CONFIG)/upmem-query-host
pack:
	rm -rf dist
	mkdir dist
	tar -zcvf dist/src.tar.gz --exclude-from=.tarignore .
push:
	scp dist/src.tar.gz upmemcloud3:~/workspace
