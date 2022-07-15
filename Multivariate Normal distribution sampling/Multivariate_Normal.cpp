#include "libInterpolate/Interpolate.hpp"
#include "libInterpolate/DataSet.hpp"
#include <fstream>
#include <iostream>
#include "eigenmvn.h"
#include <numbers>
#include <chrono>
#include "matplotlib.hpp"
#include <boost/histogram.hpp>

plot_matplotlib plot;

Eigen::MatrixXd covariance_driver();

/**
	Take a pair of un-correlated variances.
	Create a covariance matrix by correlating
	them, sandwiching them in a rotation matrix.
*/
Eigen::Matrix2d genCovar(double v0, double v1, double theta)
{
	Eigen::Matrix2d rot = Eigen::Rotation2Dd(theta).matrix();
	return rot * Eigen::DiagonalMatrix<double, 2, 2>(v0, v1) * rot.transpose();
}


int main()
{
	Eigen::Vector3d mean;
	Eigen::Matrix3d covar;
	//mean << 2, 4; // Set the mean
	mean << 0., 0, 0.; // Set the mean
	//covar << 1, 0.0, 0.0, 1;  // Set the covariance unity matrix
	//covar << 1, 0.6, 0.6, 2;  // Set the covariance
	
	Eigen::Matrix3d D, C, V; 

	D << 0.2, 0, 0, 0, 0.1, 0, 0, 0, 0.15;
	C << 1, 0.8, 0.5, 0.8, 1, 0.3, 0.50, .3, 1;
	//V = D * C * D;

	//covar = genCovar(3, 0.1, std::numbers::pi / 5.0);
	//covar = genCovar(1, 1, 1);
	//covar = genCovar(1, 0.5, 2);

	V = covariance_driver();

	const auto Samples = 10000000;
	const auto Histogram_size = 100;
	const auto Smooth_factor = 1;

	std::random_device r;
	auto seed = (uint64_t(r()) << 32) | r();
	Eigen::EigenMultivariateNormal<double> normX_cholesk(mean, V, false, seed);

	std::vector<double> xy(3 * Samples);

	auto spx = std::span(xy.begin(), Samples);
	auto spy = std::span(xy.begin() + Samples, xy.begin() + 2ULL * Samples);
	auto spz = std::span(xy.begin() + 2ULL * Samples, xy.end());

	auto begin = std::chrono::high_resolution_clock::now();

	Eigen::Map<Eigen::Matrix<double, 3, Samples>, 
		Eigen::Unaligned, Eigen::Stride<1, Samples> > k(xy.data());

	k = normX_cholesk.samples(Samples);

	auto mmx = std::ranges::minmax_element(spx);
	auto minx = *mmx.min;
	auto maxx = *mmx.max;

	auto mmy = std::ranges::minmax_element(spy);
	auto miny = *mmy.min;
	auto maxy = *mmy.max;

	auto mmz = std::ranges::minmax_element(spz);
	auto minz = *mmz.min;
	auto maxz = *mmz.max;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> fp_sec = end - begin;

	std::cout << " Number of samples = " << Samples << std::endl << std::endl
		<< " Duration EigenMultivariateNormal " << fp_sec.count() << "[s]" <<
		std::endl << std::endl;
	std::cout << " Mean " << std::endl << mean << std::endl << std::endl 
		<< " Covariance Matrix " << std::endl << V << std::endl << std::endl;

	begin = std::chrono::high_resolution_clock::now();

	auto hxy = boost::histogram::make_histogram(
		boost::histogram::axis::regular(Histogram_size, minx, maxx),
		boost::histogram::axis::regular(Histogram_size, miny, maxy),
		boost::histogram::axis::regular(Histogram_size, minz, maxz));

	auto w = { spx, spy, spz };

	hxy.fill(w);

	using namespace boost::histogram::literals; // enables _c suffix
	auto hr12 = boost::histogram::algorithm::project(hxy, 1_c, 2_c); 

	std::vector<double> X, Y, Z;

	for (auto&& x : hr12.axis(0))
		X.push_back(x);

	for (auto&& y : hr12.axis(1))
		Y.push_back(y);

	for (auto&& i : boost::histogram::indexed(hr12))
		Z.push_back(i);

	//////////////////

	auto dens_xy = Samples * abs((maxz - minz) * (maxy - miny)) / (X.size() * Y.size());

	for (auto& i : Z)
		i /= dens_xy;

	end = std::chrono::high_resolution_clock::now();
	fp_sec = end - begin;
	std::cout << " Duration boost::histogram " << fp_sec.count() << "[s]" << std::endl << std::endl;

	if (Smooth_factor != 1) {

		begin = std::chrono::high_resolution_clock::now();

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
			Histogram_size * Smooth_factor, miny, maxy),
			boost::histogram::axis::regular(Histogram_size * Smooth_factor, minz, maxz));

		for (auto&& x : h.axis(0))
			X.push_back(x);

		for (auto&& y : h.axis(1))
			Y.push_back(y);

		Z.clear();

		auto gh = 1. / Smooth_factor;
		for (auto y = 0; y < Y.size(); y++)
			for (auto x = 0; x < X.size(); x++)
				Z.push_back(interp2d(double(x) * gh, double(y) * gh));

		end = std::chrono::high_resolution_clock::now();
		fp_sec = end - begin;
		std::cout << " Duration BicubicInterpolator " << fp_sec.count() << "[s]" << std::endl;
	}

	begin = std::chrono::high_resolution_clock::now();
	plot.plot_histogram(X, Y, Z);

	plot.set_xlabel("X");
	plot.set_ylabel("Y");
	plot.set_zlabel("p(X)");
	plot.grid_on();
	plot.set_title("Bivariate normal distribution sampling, joined density");
	
	end = std::chrono::high_resolution_clock::now();
	fp_sec = end - begin;
	std::cout << " Duration plot_histogram " << fp_sec.count() << "[s]" << std::endl << std::endl;

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