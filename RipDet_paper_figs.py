epochs = 10
G_s = 1
W_s = 2459 * 2
alpha_1 = 1e-4
for i in range(0, epochs):
	
	for j in range(0, 2459):
		G_s = G_s + 1

		if G_s < W_s:
			lr = G_s / W_s * alpha_1

		print(lr)


