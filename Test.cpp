// -----------------------------------------------------------------------------
// MIT License
//
// Copyright (c) 2020 Rui LIU (@liurui39660)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// -----------------------------------------------------------------------------
#include <random>
#include "AUROC.hpp"

int main(int argc, char* argv[]) {
	std::random_device eng;
	std::uniform_real_distribution<> random;
	std::normal_distribution<> randn;
	std::uniform_int_distribution<> randint(0, 1);

	const auto n = 100000; // How many records?
	const auto acc = 1 - random(eng) / 3; // (TP + TN) / (P + N)
	const auto scale = 5; // Separate pos and neg

	const auto score = new double[n];
	const auto label = new double[n];

	const auto fileLabel = fopen(SOLUTION_DIR"Label.txt", "wb");
	const auto fileScore = fopen(SOLUTION_DIR"Score.txt", "wb");
	for (int i = 0; i < n; i++) {
		fprintf(fileLabel, "%g\n", label[i] = randint(eng));
		fprintf(fileScore, "%f\n", score[i] = random(eng) < acc ? scale * label[i] + randn(eng) : scale * !label[i] + randn(eng));
	}
	fclose(fileLabel);
	fclose(fileScore);

	printf("AUROC.hpp:\t%.16f\n", AUROC(label, score, n));
	delete[] score;
	delete[] label;

	char cmd[1024];
	sprintf(cmd, "python %s %s %s", SOLUTION_DIR"AUROC.py", SOLUTION_DIR"Label.txt", SOLUTION_DIR"Score.txt");
	system(cmd);
}
