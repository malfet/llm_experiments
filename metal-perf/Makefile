all: int8mm int4mm triu_indices

int8mm: int8mm.mm
	clang++ -O3 -std=c++17  -Wall -Wextra -o $@ $< -framework Metal -framework Foundation

int4mm: int4mm.mm
	clang++ -O3 -std=c++17  -Wall -Wextra -o $@ $< -framework Metal -framework Foundation

trilu_indices.metallib: trilu_indices.metal
	xcrun metal -o $@ $<

triu_indices: triu_indices.mm trilu_indices.metallib
	clang++ -g -std=c++17  -Wall -Wextra -o $@ triu_indices.mm -framework Metal -framework Foundation -Wl,-sectcreate,__TEXT,metal_library, trilu_indices.metallib

run: int8mm int4mm
	./int8mm
	./int4mm
