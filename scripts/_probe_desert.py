from bgai.engine.tm.board import base_board, hex_distance

b = base_board()
deserts = ["A5", "B1", "B4", "B6", "D3", "E8", "F3", "G4", "G7", "H1", "I7"]
for d in deserts:
    land = [(n, b.hexes[n].color) for n in sorted(b.adjacent[d]) if not n.startswith("r")]
    print(d, "->", land)
print("--- desert pair distances <=3 ---")
for i, a in enumerate(deserts):
    for c in deserts[i + 1:]:
        dist = hex_distance(a, c)
        if dist <= 3:
            print(a, c, dist)
