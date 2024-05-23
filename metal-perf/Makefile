all: int8mm int4mm

int8mm: int8mm.mm
	clang++ -O3 -std=c++17  -Wall -Wextra -o $@ $< -framework Metal -framework Foundation

int4mm: int4mm.mm
	clang++ -O3 -std=c++17  -Wall -Wextra -o $@ $< -framework Metal -framework Foundation

run: int8mm int4mm
	./int8mm
	./int4mm
