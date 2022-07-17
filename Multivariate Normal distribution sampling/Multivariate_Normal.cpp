#include "libInterpolate/Interpolate.hpp"
#include "libInterpolate/DataSet.hpp"
#include "eigenmvn.h"
#include <numbers>
#include <chrono>
#include "matplotlib.hpp"
#include <boost/histogram.hpp>

plot_matplotlib plot;

Eigen::MatrixXd covariance_driver();

int main()
{
	const bool Sine = true;
	const auto Samples = 100000;
	const auto integ = 1000;
	const auto Histogram_size = 100;
	const auto Smooth_factor = 1;

	Eigen::Vector3d mean;
	Eigen::Matrix3d covar;

	mean << 0, 0, 0; // Set the mean
	//covar << 1, 0.6, 0.6, 2;  

	//covar = covar.Identity();
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

	Eigen::Map<Eigen::Matrix<double, 3, Samples>,
		Eigen::Unaligned, Eigen::Stride<1, Samples> > k(xy.data());

	std::vector<double> Z2(Histogram_size * Histogram_size);
	std::vector<double> X, Y, Z;

	double minx = INFINITY, maxx = -INFINITY, miny = INFINITY, maxy = -INFINITY,
		minz = INFINITY, maxz = -INFINITY;
	double minxo, maxxo, minyo, maxyo, minzo, maxzo;

	std::chrono::time_point<std::chrono::high_resolution_clock> begin, end;
	std::chrono::duration<double> fp_sec;

	for (auto i = 0; i < integ; i++) {

		begin = std::chrono::high_resolution_clock::now();

		k = normX_cholesk.samples(Samples);
		
		minxo = minx, maxxo = maxx, minyo = miny, maxyo = maxy, minzo = minz, maxzo = maxz;

		auto mmx = std::ranges::minmax_element(spx);
		minx = *mmx.min;
		maxx = *mmx.max;

		auto mmy = std::ranges::minmax_element(spy);
		miny = *mmy.min;
		maxy = *mmy.max;

		auto mmz = std::ranges::minmax_element(spz);
		minz = *mmz.min;
		maxz = *mmz.max;

		end = std::chrono::high_resolution_clock::now();
		fp_sec = end - begin;

		if (i == 0)
			std::cout << std::endl << " Duration EigenMultivariateNormal "
			<< fp_sec.count() << "[s]" << std::endl << std::endl;

		begin = std::chrono::high_resolution_clock::now();

		using namespace boost::histogram::literals; // enables _c suffix

		if (minxo < minx)
			minx = minxo;
		
		if(maxxo > maxx)
			maxx = maxxo;

		if (minyo < miny)
			miny = minyo;

		if (maxyo > maxy)
			maxy = maxyo;

		if (minzo < minz)
			minz = minzo;

		if (maxzo > maxz)
			maxz = maxzo;
		
		auto hxy = boost::histogram::make_histogram(
			boost::histogram::axis::regular(Histogram_size, minx, maxx),
			boost::histogram::axis::regular(Histogram_size, miny, maxy),
			boost::histogram::axis::regular(Histogram_size, minz, maxz));

		auto w = { spx, spy, spz };

		hxy.fill(w);
		
		auto hr12 = boost::histogram::algorithm::project(hxy, 1_c, 2_c);

		X.clear();
		Y.clear();
		Z.clear();

		for (auto && y : hr12.axis(0)) 
				X.push_back(y);
		
		for (auto&& z : hr12.axis(1))
			Y.push_back(z);

		for (auto&& i : boost::histogram::indexed(hr12))
			Z.push_back(i);
		
		auto dens_xy = Samples * abs((maxy - miny) * (maxz - minz)) / (X.size() * Y.size());

		for (auto t = 0; auto & i : Z) {
			i /= dens_xy;
			Z2[t] += i / integ;
			t++;
		}

		end = std::chrono::high_resolution_clock::now();
		fp_sec = end - begin;
		if (i == 0)
			std::cout << " Duration boost::histogram " << fp_sec.count()
			<< "[s]" << std::endl << std::endl;

		if (Smooth_factor != 1 && i == integ - 1) {

			begin = std::chrono::high_resolution_clock::now();

			_2D::BicubicInterpolator<double> interp2d;

			_2D::DataSet data(X.size() * Smooth_factor, Y.size() * Smooth_factor);

			std::vector<double> Zi(data.x.size());

			auto nx = X.size() * Smooth_factor;
			auto ny = Y.size() * Smooth_factor;

			for (auto i = 0; i < nx; i++)
				for (auto j = 0; j < ny; j++)
					Zi[i * ny + j] = Z2[(j + i * X.size()) % Z2.size()];

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

			Z2.clear();

			auto gh = 1. / Smooth_factor;
			for (auto y = 0; y < Y.size(); y++)
				for (auto x = 0; x < X.size(); x++)
					Z2.push_back(interp2d(double(x) * gh, double(y) * gh));

			end = std::chrono::high_resolution_clock::now();
			fp_sec = end - begin;
			if (i == 0)
				std::cout << " Duration BicubicInterpolator " << fp_sec.count()
				<< "[s]" << std::endl;
		}
	}

	begin = std::chrono::high_resolution_clock::now();

	plot.plot_histogram(X, Y, Z2, Sine);

	plot.set_xlabel("X");
	plot.set_ylabel("Y");
	plot.set_zlabel("p(X)");
	plot.grid_on();
	if(!Sine)
		plot.set_title("Bivariate normal distribution, joined density");
	else 
		plot.set_title("Bivariate sine distribution, joined density");

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
