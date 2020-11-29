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
#pragma once

#include <algorithm>

template<class T>
double AUROC(const T* y_true, const T* y_pred, size_t n) {
	for (size_t i = 0; i < n; i++) {
		if (std::isnan(y_pred[i]) || std::isinf(y_pred[i]))
			return -1;
	}

	const auto index = new size_t[n];
	for (size_t i = 0; i < n; i++) index[i] = i;
	// std::iota(index, index + n, 0); // Equivalent to above
	std::sort(index, index + n, [&](size_t a, size_t b) { return y_pred[a] > y_pred[b]; });
	const auto y = new double[n]; // Desc
	const auto z = new double[n]; // Desc
	for (size_t i = 0; i < n; i++) {
		y[i] = y_true[index[i]];
		z[i] = y_pred[index[i]];
	}

	const auto tp = y; // Reuse
	tp[0] = y[0];
	for (size_t i = 1; i < n; i++) {
		tp[i] = tp[i - 1] + y[i];
	} // Equivalent to below
	// std::partial_sum(y, y + n, tp);

	size_t top = 0; // # diff
	for (size_t i = 0; i < n - 1; i++) {
		if (z[i] != z[i + 1])
			index[top++] = i;
	}
	index[top++] = n - 1;
	n = top; // Size of y/z -> sizeof tps/fps
	delete[] z;

	const auto fp = new double[n];
	for (size_t i = 0; i < n; i++) {
		tp[i] = tp[index[i]];
		fp[i] = 1 + index[i] - tp[i];
	}
	delete[] index;

	const auto tps_diff = new double[n];
	const auto fps_diff = new double[n];
	for (size_t i = 1; i < n; i++) {
		tps_diff[i] = tp[i] - tp[i - 1];
		fps_diff[i] = fp[i] - fp[i - 1];
	} // Equivalent to below
	// std::adjacent_difference(tp, tp + n, tps_diff);
	// std::adjacent_difference(fp, fp + n, fps_diff);
	top = 1;
	for (size_t i = 1; i < n - 1; i++) {
		if (tps_diff[i] != tps_diff[i + 1] || fps_diff[i] != fps_diff[i + 1]) {
			tp[top] = tp[i];
			fp[top] = fp[i];
			top++;
		}
	}
	tp[top] = tp[n - 1];
	fp[top] = fp[n - 1];
	n = ++top; // Size of tp/fp -> size of optimized tp/fp 
	delete[] tps_diff;
	delete[] fps_diff;

	for (size_t i = 0; i < n; i++) {
		tp[i] /= tp[n - 1];
		fp[i] /= fp[n - 1];
	}

	double area = tp[0] * fp[0] / 2; // The first triangle from origin
	for (size_t i = 0; i < n - 1; i++) {
		area += (tp[i] + tp[i + 1]) * (fp[i + 1] - fp[i]) / 2;
	}

	delete[] tp;
	delete[] fp;

	return area;
}
