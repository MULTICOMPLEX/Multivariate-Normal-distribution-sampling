#ifndef testing_NoniusBenchmarks_Utils_DataSet_hpp
#define testing_NoniusBenchmarks_Utils_DataSet_hpp

/** @file DataSet.hpp
	* @brief
	* @author C.D. Clark III
	* @date 06/11/17
	*/

namespace _2D {

	class DataSet
	{
	public:
		std::vector<double> x, y, z;
		DataSet(auto nx = 3, auto ny = 3)
		{
			x = std::vector<double>(nx * ny);
			y = std::vector<double>(nx * ny);
			//z = std::vector<double>(nx*ny);
			for (auto i = 0; i < nx; i++)
			{
				for (auto j = 0; j < ny; j++)
				{
					auto k = i * ny + j;
					x[k] = i;
					y[k] = j;
					//z[k] = k;
				}
			}
		}

	};

}


#endif // include protector
