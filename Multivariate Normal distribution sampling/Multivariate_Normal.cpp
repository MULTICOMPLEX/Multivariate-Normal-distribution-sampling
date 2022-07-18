#include "libInterpolate/Interpolate.hpp"
#include "libInterpolate/DataSet.hpp"
#include "eigenmvn.h"
#include <numbers>
#include <chrono>
#include <ranges>
#include "matplotlib.hpp"
#include <boost/histogram.hpp>
using namespace boost::histogram::literals; // enables _c suffix

plot_matplotlib plot;

Eigen::MatrixXd covariance_driver();

template <typename T>
void normalize_vector
(
	std::vector<T>& v,
	const T a,
	const T b);

template <typename T>
void null_offset_vector(std::vector<T>& v);

void BicubicInterpolator(auto Smooth_factor, auto Histogram_size,
	auto& X, auto& Y, auto& Z,
	const double minx, const double maxx, const double miny, const double maxy);

int main()
{
	const bool Sine = true;
	const auto Samples = 10000;
	const auto Integrations = 1000;
	const auto Histogram_size = 50;
	const auto Smooth_factor = 1;

	Eigen::Vector3d mean;
	Eigen::Matrix3d covar;

	mean << 0, 0, 0; // Set the mean
	//covar << 1, 0.6, 0.6, 2;  

	covar = covar.Identity();
	covar = covariance_driver();

	std::cout << " Number of samples = " << Samples << std::endl << std::endl;
	std::cout << " Mean " << std::endl << mean << std::endl << std::endl
		<< " Covariance Matrix " << std::endl << covar << std::endl;

	std::random_device r;
	auto seed = (uint64_t(r()) << 32) | r();
	Eigen::EigenMultivariateNormal<double> normX_cholesk(mean, covar, false, seed, Sine);

	std::vector<double> xy(3 * Samples);

	auto spx = std::span(xy.begin(), Samples);
	auto spy = std::span(xy.begin() + Samples, xy.begin() + 2ULL * Samples);
	auto spz = std::span(xy.begin() + 2ULL * Samples, xy.end());

	auto spanxy = { spx, spy, spz };

	Eigen::Map<Eigen::Matrix<double, 3, Samples>,
		Eigen::Unaligned, Eigen::Stride<1, Samples> > multivariate_data_buffer(xy.data());

	std::vector<double> Z(Histogram_size * Histogram_size);
	std::vector<double> X, Y;

	double minx = INFINITY, maxx = -INFINITY, miny = INFINITY, maxy = -INFINITY,
		minz = INFINITY, maxz = -INFINITY;
	double minxo, maxxo, minyo, maxyo, minzo, maxzo;

	std::chrono::duration<double> fp1_sec = {}, fp2_sec = {};

	mxws<uint32_t> rng;

	for (auto i = 0; i < Integrations; i++) {

		if (i == 0) {
			std::cout << std::endl << " busy..." << std::endl << std::endl;
		}

		auto begin = std::chrono::high_resolution_clock::now();

		if (!Sine) {

			multivariate_data_buffer = normX_cholesk.samples(Samples);

			minxo = minx, maxxo = maxx, minyo = miny, maxyo = maxy, minzo = minz, maxzo = maxz;

			auto mmx = std::ranges::minmax_element(spx);
			minx = *mmx.min; maxx = *mmx.max;

			auto mmy = std::ranges::minmax_element(spy);
			miny = *mmy.min; maxy = *mmy.max;

			auto mmz = std::ranges::minmax_element(spz);
			minz = *mmz.min; maxz = *mmz.max;

			if (minxo < minx) minx = minxo; if (maxxo > maxx) maxx = maxxo;
			if (minyo < miny) miny = minyo; if (maxyo > maxy) maxy = maxyo;
			if (minzo < minz) minz = minzo; if (maxzo > maxz) maxz = maxzo;
		}

		else {
			std::ranges::for_each(xy, [&](auto& r) {r = rng.sine<double>(); });
			minx = 0, miny = 0, minz = 0;
			maxx = rng.board_SIZE, maxy = rng.board_SIZE, maxz = rng.board_SIZE;
		}

		auto end = std::chrono::high_resolution_clock::now();

		fp1_sec += end - begin;

		if (i == Integrations - 1) {
			std::cout << std::endl << " Duration EigenMultivariateNormal "
				<< fp1_sec.count() << "[s]" << std::endl << std::endl;
		}

		begin = std::chrono::high_resolution_clock::now();

		auto hxy = boost::histogram::make_histogram(
			boost::histogram::axis::regular(Histogram_size, minx, maxx),
			boost::histogram::axis::regular(Histogram_size, miny, maxy),
			boost::histogram::axis::regular(Histogram_size, minz, maxz));

		hxy.fill(spanxy);

		auto hr12 = boost::histogram::algorithm::project(hxy, 1_c, 2_c);

		auto dens_xy = Samples * abs((maxy - miny) * (maxz - minz)) /
			(hr12.axis(0).size() * hr12.axis(1).size());

		for (auto t = 0; auto && i : boost::histogram::indexed(hr12)) {
			Z[t] += i / Integrations / dens_xy;
			t++;
		}

		end = std::chrono::high_resolution_clock::now();

		fp2_sec += end - begin;

		if (i == Integrations - 1) {
			
			std::cout << " Duration boost::histogram " << fp2_sec.count()
				<< "[s]" << std::endl << std::endl;

			for (auto&& y : hr12.axis(0))
				X.push_back(y);

			for (auto&& z : hr12.axis(1))
				Y.push_back(z);
		}
	}

	if (Smooth_factor != 1) {

		auto begin = std::chrono::high_resolution_clock::now();

		BicubicInterpolator(Smooth_factor, Histogram_size,
			X, Y, Z, miny, maxy, minz, maxz);

		auto end = std::chrono::high_resolution_clock::now();
		fp1_sec = end - begin;

		std::cout << std::endl << " Duration BicubicInterpolator " << fp1_sec.count()
			<< "[s]" << std::endl;
	}

	auto begin = std::chrono::high_resolution_clock::now();

	if (Sine) {
		normalize_vector(Z, -1., 1.);
		null_offset_vector(Z);
	}

	plot.plot_histogram(X, Y, Z, Sine);

	plot.set_xlabel("X");
	plot.set_ylabel("Y");
	plot.set_zlabel("p(X)");
	plot.grid_on();
	if (!Sine)
		plot.set_title("Bivariate normal distribution, joined density");
	else
		plot.set_title("Sine distribution, joined density");

	auto end = std::chrono::high_resolution_clock::now();
	fp1_sec = end - begin;
	std::cout << " Duration plot_histogram " << fp1_sec.count() << "[s]" << std::endl << std::endl;

	plot.show();
}

Eigen::MatrixXd covariance_driver()
{
	Eigen::Matrix3d mat;

	mat << 0.4, 1.16, 0.15,
		0.16, 0.01, 0.45,
		-0.15, 0.45, 1.0225;

	//auto x_mean = mat.colwise().mean();
	//Eigen::Matrix3d cov = ((mat.rowwise() - x_mean).matrix().transpose()
		//* (mat.rowwise() - x_mean).matrix()) / (mat.rows() - 1);

	Eigen::MatrixXd centered = mat.rowwise() - mat.colwise().mean();
	Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(mat.rows() - 1);

	return cov;
}

template <typename T>
void normalize_vector
(
	std::vector<T>& v,
	const T a,
	const T b)
{
	auto k = std::ranges::minmax_element(v);
	auto min = *k.min;
	auto max = *k.max;

	auto normalize = [&](auto& n) {n = a + (n - min) * (b - a) / (max - min); };

	std::ranges::for_each(v, normalize);
}

template <typename T>
void null_offset_vector(std::vector<T>& v)
{
	T mean = 0;
	for (auto& d : v)
		mean += d;

	mean /= v.size();

	for (auto& d : v)
		d -= mean;
}

void BicubicInterpolator(auto Smooth_factor, auto Histogram_size,
	auto& X, auto& Y, auto& Z,
	const double minx, const double maxx, const double miny, const double maxy)
{
	_2D::BicubicInterpolator<double> interp2d;

	_2D::DataSet data(X.size() * Smooth_factor, Y.size() * Smooth_factor);

	std::vector<double> Zi(data.x.size());

	auto nx = X.size() * Smooth_factor;
	auto ny = Y.size() * Smooth_factor;

	for (auto i = 0; i < nx; i++)
		for (auto j = 0; j < ny; j++)
			Zi[i * ny + j] = Z[(j + i * X.size()) % Z.size()];

	interp2d.setData(data.x, data.y, Zi);

	X.clear();
	Y.clear();

	auto h = boost::histogram::make_histogram(boost::histogram::axis::regular(
		Histogram_size * Smooth_factor, minx, maxx),
		boost::histogram::axis::regular(Histogram_size * Smooth_factor, miny, maxy));

	for (auto&& x : h.axis(0))
		X.push_back(x);

	for (auto&& y : h.axis(1))
		Y.push_back(y);

	Z.clear();

	auto gh = 1. / Smooth_factor;
	for (auto y = 0; y < Y.size(); y++)
		for (auto x = 0; x < X.size(); x++)
			Z.push_back(interp2d(double(x) * gh, double(y) * gh));
}
