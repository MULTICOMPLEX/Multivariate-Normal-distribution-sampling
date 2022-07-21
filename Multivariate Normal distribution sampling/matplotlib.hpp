#pragma once

#include <Python.h>
#include <vector>
#include <string>

class plot_matplotlib
{

private:

	void PyRun_StringStd(std::string somestring);
	bool _autorange;
	std::string _pythoncmd;
	double _range_x1, _range_x2,
		_range_y1, _range_y2,
		_range_z1, _range_z2; // Range of the plot

public:

	static std::vector<std::string> colors;

	plot_matplotlib(); // Constructor

	~plot_matplotlib(); // Destructor

	void init_plot_window(const char* name, int x, int y);

	//Set the label on the X axis
	void set_xlabel(std::string xlabel, std::string properties = "");

	//Set the label on the Y axis
	void set_ylabel(std::string ylabel, std::string properties = "");

	//Set the label on the Z axis	
	void set_zlabel(std::string zlabel, std::string properties = "");

	//Set the plot title
	void set_title(std::string title, std::string font = "DejaVu Sans", int fs = 10);

	//Set the aspect ratio of the plot to equal (desired for plotting paths and path Segments)
	void set_equal_ascpectratio();

	//Set a figure (required for using subplots)
	void figure(int fignumber);

	//Set a subplot number
	void subplot(int plotnumber);

	//Set the X and Y range of the plot
	template <typename T>
	void set_xyrange(T x1, T x2, T y1, T y2)
	{
		_autorange = false;
		PyRun_StringStd("plt.axis([" + std::to_string(x1) + ", " + std::to_string(x2) + ", " + std::to_string(y1) + ", " + std::to_string(y2) + "])");
	}

	template <typename T>
	void set_xyzrange(T x1, T x2, T y1, T y2, T z1, T z2)
	{
		_autorange = false;
		PyRun_StringStd("plt.axis(["
			+ std::to_string(x1) + ", " + std::to_string(x2) +
			", " + std::to_string(y1) + ", " + std::to_string(y2) +
			", " + std::to_string(z1) + ", " + std::to_string(z2) +
			"])");
	}

	//Enable a legend
	void enable_legend();

	//Calculate the plot range dependent on the added data (all data visible + small border, better than built standard range)
	void set_range_auto();

	void run_customcommand(std::string command);

	void adjust_ticker();

	void show();

	//Add data to the plot with markers (X/Y points and matplotlib properties, e.g. 'o' for points or no properties for points connected by lines)

	void plot_somedata(const auto& X, const auto& Y,
		std::string properties = "k", std::string label = "Line 1", std::string color = "green",
		double linewidth = 2, double markersize = 1, double alpha = 1);

	void plot_somedata_step(const std::vector<double>& X, const std::vector<double>& Y,
		std::string properties = "k", std::string label = "Line 1", std::string color = "green",
		double linewidth = 2, double markersize = 1, double alpha = 1);

	void plot_somedata_3D(const std::vector<double>& X, const std::vector<double>& Y, const std::vector<double>& Z,
		std::string properties = "k", std::string label = "Line 1", std::string color = "green", double alpha = 1);

	void plot_histogram(const auto& X, const auto& Y, const auto& Z, bool Sine);

	void grid_off();
	void grid_on();
	void line(const double x1 = 0, const double y1 = 0, const double x2 = 10, const double y2 = 10,
		const std::string color = "black", const double linewidth = 2,
		const std::string linestyle = "solid");

	void text(const double x, const double y,
		std::string somestring, std::string color, double fontsize);

	void arrow(const double x_tail, const double y_tail,
		const double x_head, const double y_head, std::string color);

	void imshow(const std::string& points, const std::string& cmap = "gray", double extent = 1.0);
};


void plot_matplotlib::PyRun_StringStd(std::string somestring)
{
	PyRun_SimpleStringFlags(somestring.c_str(), NULL);
	if (_pythoncmd != "") {
		_pythoncmd += "\n";
	}
	_pythoncmd += somestring;
}

plot_matplotlib::plot_matplotlib() // Constructor
{
	//python -m pip install pyqt5

	Py_Initialize();

	// Initialize automatic range:
	_range_x1 = DBL_MAX;
	_range_x2 = DBL_MIN;
	_range_y1 = DBL_MAX;
	_range_y2 = DBL_MIN;
	_range_z1 = DBL_MAX;
	_range_z2 = DBL_MIN;

	_pythoncmd = "";
	_autorange = true;

	PyRun_StringStd("import matplotlib");

	PyRun_StringStd("import matplotlib.pyplot as plt");

	PyRun_StringStd("import matplotlib.ticker as ticker");
	PyRun_StringStd("import numpy as np");
	PyRun_StringStd("from mpl_toolkits.mplot3d import Axes3D");
	PyRun_StringStd("from matplotlib import cm");
	PyRun_StringStd("import math");
}

plot_matplotlib::~plot_matplotlib() // Destructor
{
	Py_Finalize();
	//cout << "- Python plot command:\n";
	//cout << pythoncmd;
}

void plot_matplotlib::grid_on()
{
	PyRun_StringStd("plt.grid(True)");
}

void plot_matplotlib::grid_off()
{
	PyRun_StringStd("plt.grid(False)");
}


//Set the label on the X axis
void plot_matplotlib::set_xlabel(std::string xlabel, std::string properties)
{
	if (properties != "") {
		properties = ", " + properties;
	}
	PyRun_StringStd("plt.xlabel('" + xlabel + "'" + properties + ")");
}
//Set the label on the Y axis
void plot_matplotlib::set_ylabel(std::string ylabel, std::string properties)
{
	if (properties != "") {
		properties = ", " + properties;
	}
	PyRun_StringStd("plt.ylabel('" + ylabel + "'" + properties + ")");
}

//Set the label on the z axis
void plot_matplotlib::set_zlabel(std::string zlabel, std::string properties)
{
	if (properties != "") {
		properties = ", " + properties;
	}
	PyRun_StringStd("ax.set_zlabel('" + zlabel + "'" + properties + ")");
}

//Set the plot title
void plot_matplotlib::set_title(std::string title, std::string font, int fs)
{
	title.append(R"(")");
	title.insert(0, R"(")");

	PyRun_StringStd(
		"plt.title(" + title + ", font='" + font + "', fontsize="
		+ std::to_string(fs) + " )");
}

//Set the aspect ratio of the plot to equal (desired for plotting paths and path Segments)
void plot_matplotlib::set_equal_ascpectratio()
{
	PyRun_StringStd("plt.axes().set_aspect('equal')");
}

//Set a figure (required for using subplots)
void plot_matplotlib::figure(int fignumber)
{
	PyRun_StringStd("plt.figure(" + std::to_string(fignumber) + ")");
}
//Set a subplot number
void plot_matplotlib::subplot(int plotnumber)
{
	PyRun_StringStd("plt.subplot(" + std::to_string(plotnumber) + ")");
}

//Enable a legend
void plot_matplotlib::enable_legend()
{
	PyRun_StringStd("plt.legend()");
}

//Calculate the plot range dependent on the added data (all data visible + small border, better than built standard range)
void plot_matplotlib::set_range_auto()
{
	double extendpercent = 0.04f;
	double xrange = _range_x2 - _range_x1;
	double yrange = _range_y2 - _range_y1;
	double zrange = _range_z2 - _range_z1;

	if (xrange < 0) {
		xrange = xrange * -1.0;
	}
	if (yrange < 0) {
		yrange = yrange * -1.0;
	}

	if (zrange < 0) {
		zrange = zrange * -1.0;
	}

	set_xyrange(_range_x1 - xrange * extendpercent, _range_x2 + xrange * extendpercent,
		_range_y1 - yrange * extendpercent, _range_y2 + yrange * extendpercent);

	set_xyzrange(_range_x1 - xrange * extendpercent, _range_x2 + xrange * extendpercent,
		_range_y1 - yrange * extendpercent, _range_y2 + yrange * extendpercent,
		_range_z1 - zrange * extendpercent, _range_z2 + zrange * extendpercent
	);
}

void plot_matplotlib::run_customcommand(std::string command)
{
	PyRun_StringStd("plt." + command);
}

void plot_matplotlib::adjust_ticker()
{
	PyRun_StringStd("ax = plt.gca()"); // to get current axis
	PyRun_StringStd("ax.xaxis.set_major_locator(ticker.AutoLocator())");
	PyRun_StringStd("ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())");
	PyRun_StringStd("ax.yaxis.set_major_locator(ticker.AutoLocator())");
	PyRun_StringStd("ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())");
	PyRun_StringStd("ax.tick_params(which = 'both', direction='in')");
}

//Finally show the plot window (interrupts program execution until plot window is closed)

void plot_matplotlib::show()
{
	//if(autorange && !_autorange) { this->set_range_auto(); }
	//if(keepaspectratio) { this->set_equal_ascpectratio(); }
	PyRun_StringStd("plt.show()");
}

void plot_matplotlib::plot_somedata(const auto& X, const auto& Y,
	std::string properties, std::string label, std::string color,
	double linewidth, double markersize, double alpha)
{
	// Plot Points:
	std::string xpoints = "";
	std::string ypoints = "";

	properties.append("'");
	properties.insert(0, "'");
	label.append("'");
	label.insert(0, "'");
	color.append("'");
	color.insert(0, "'");

	for (size_t i = 0; i < X.size(); i++) {
		// Add points
		if (i > 0) {
			xpoints += ",";
		}
		xpoints += std::to_string(X[i]);

		if (i > 0) {
			ypoints += ",";
		}
		ypoints += std::to_string(Y[i]);

		// Set auto range
		if (X[i] < _range_x1) {
			_range_x1 = X[i];
		}
		if (X[i] > _range_x2) {
			_range_x2 = X[i];
		}

		if (Y[i] < _range_y1) {
			_range_y1 = Y[i];
		}
		if (Y[i] > _range_y2) {
			_range_y2 = Y[i];
		}
	}

	if (properties != "") {
		properties = ", " + properties;
	}

	PyRun_StringStd(
		"plt.plot( [" + xpoints + "], [" + ypoints + "]" + properties + ",label=" + label + ",\
color=" + color + ",\
linewidth=" + std::to_string(linewidth) + ", markersize=" + std::to_string(markersize) + ",\
alpha=" + std::to_string(alpha) + ")");

	if (label != "")PyRun_StringStd("plt.legend()");
}

//pip install pycairo
//python -m pip install -U pip
//python - m pip install - U matplotlib

void plot_matplotlib::plot_somedata_step(const std::vector<double>& X, const std::vector<double>& Y,
	std::string properties, std::string label, std::string color, double linewidth, double markersize, double alpha)
{
	// Plot Points:
	std::string xpoints = "";
	std::string ypoints = "";

	properties.append("'");
	properties.insert(0, "'");
	label.append("'");
	label.insert(0, "'");
	color.append("'");
	color.insert(0, "'");

	for (size_t i = 0; i < X.size(); i++) {
		// Add points
		if (i > 0) {
			xpoints += ",";
		}
		xpoints += std::to_string(X.at(i));

		if (i > 0) {
			ypoints += ",";
		}
		ypoints += std::to_string(Y.at(i));

		// Set auto range
		if (X.at(i) < _range_x1) {
			_range_x1 = X.at(i);
		}
		if (X.at(i) > _range_x2) {
			_range_x2 = X.at(i);
		}

		if (Y.at(i) < _range_y1) {
			_range_y1 = Y.at(i);
		}
		if (Y.at(i) > _range_y2) {
			_range_y2 = Y.at(i);
		}
	}

	if (properties != "") {
		properties = ", " + properties;
	}

	PyRun_StringStd(
		"plt.step( [" + xpoints + "], [" + ypoints + "]" + properties + ", label=" + label + ",color=" + color + ",\
      linewidth=" + std::to_string(linewidth) + ", markersize=" + std::to_string(markersize) + ",\
      alpha=" + std::to_string(alpha) + ")");

	if (label != "")PyRun_StringStd("plt.legend()");
}

void plot_matplotlib::imshow(const std::string& points, const std::string& cmap, double extent)
{
	// Plot Points:
	std::string str;

	str.reserve(points.size());
	for (auto& i : points)
	{
		str += i;
		if (i == ']')
			str += ',';
	}

	str.erase(std::prev(str.end() - 1));

	PyRun_StringStd("E = " + std::to_string(extent) + " ");
	PyRun_StringStd("plt.imshow(" + str + ", cmap = '" + cmap + "', extent=[-E, E, -E, E])");

}

void plot_matplotlib::plot_somedata_3D(const std::vector<double>& X, const std::vector<double>& Y,
	const std::vector<double>& Z, std::string properties,
	std::string label, std::string color, double alpha)
{
	// Plot Points:
	std::string xpoints = "";
	std::string ypoints = "";
	std::string zpoints = "";

	properties.append("'");
	properties.insert(0, "'");
	label.append("'");
	label.insert(0, "'");
	color.append("'");
	color.insert(0, "'");

	for (size_t i = 0; i < X.size(); i++) {
		// Add points
		if (i > 0) {
			xpoints += ",";
		}
		xpoints += std::to_string(X.at(i));

		if (i > 0) {
			ypoints += ",";
		}
		ypoints += std::to_string(Y.at(i));

		if (i > 0) {
			zpoints += ",";
		}
		zpoints += std::to_string(Z.at(i));

		// Set auto range
		if (X.at(i) < _range_x1) {
			_range_x1 = X.at(i);
		}
		if (X.at(i) > _range_x2) {
			_range_x2 = X.at(i);
		}

		if (Y.at(i) < _range_y1) {
			_range_y1 = Y.at(i);
		}
		if (Y.at(i) > _range_y2) {
			_range_y2 = Y.at(i);
		}

		if (Z.at(i) < _range_z1) {
			_range_z1 = Z.at(i);
		}
		if (Z.at(i) > _range_z2) {
			_range_z2 = Z.at(i);
		}
	}

	if (properties != "") {
		properties = ", " + properties;
	}

	//PyRun_StringStd("fig = plt.figure()");
	PyRun_StringStd("ax = plt.subplot(111, projection = '3d')");

	PyRun_StringStd(
		"ax.plot( [" + xpoints + "], [" + ypoints + "], [" + zpoints + "]" + properties + ",\
	label=" + label + ",color=" + color + ",alpha=" + std::to_string(alpha) + ")");

	if (label != "")PyRun_StringStd("plt.legend()");
}

void plot_matplotlib::plot_histogram
(
	const auto& X, const auto& Y, const auto& Z, bool Sine
)
{
	// Plot Points:
	std::string xpoints;
	std::string ypoints;
	std::string zpoints;

	for (auto i = 0; i < X.size(); i++)
		// Add points
		xpoints.append(std::to_string(X[i]) + ",");

	for (auto i = 0; i < Y.size(); i++)
		// Add points
		ypoints.append(std::to_string(Y[i]) + ",");

	for (auto i = 0; i < Z.size(); i++)
		// Add points
		zpoints.append(std::to_string(Z[i]) + ",");

	PyRun_StringStd("hx = np.array([" + xpoints + "])");
	PyRun_StringStd("hy = np.array([" + ypoints + "])");
	PyRun_StringStd("hzz = np.array([" + zpoints + "])");

	PyRun_StringStd("hz = np.reshape(hzz, (hx.size, hy.size))");

	PyRun_StringStd("fig = plt.figure()");
	PyRun_StringStd("ax = fig.add_subplot(projection = '3d')");

	PyRun_StringStd("X, Y = np.meshgrid(hx, hy)");

	PyRun_StringStd("cp = ax.plot_surface(X, Y, hz, cmap = cm.viridis, antialiased=True)");
	PyRun_StringStd("maxz = np.max(hz)");

	if (!Sine) {
		PyRun_StringStd("csetx = ax.contourf(X, Y, hz, 60, zdir = 'x', offset = np.min(hx),\
 cmap = cm.viridis, antialiased=True)");
		PyRun_StringStd("csety = ax.contourf(X, Y, hz, 60, zdir = 'y', offset = np.max(hy),\
 cmap = cm.viridis, antialiased=True)");
		PyRun_StringStd("cset = ax.contourf\
		(X, Y, hz, 10, zdir = 'z', offset = -maxz/2, cmap = cm.viridis, antialiased=True)");
		PyRun_StringStd("ax.set_zlim(-maxz/2, maxz)");
	}

	else {
		PyRun_StringStd("cset = ax.contourf\
		(X, Y, hz, 10, zdir = 'z', offset = -1.5, cmap = cm.viridis, antialiased=True)");
		PyRun_StringStd("ax.set_zlim(-1.5, 1.0)");
	}

	PyRun_StringStd("plt.colorbar(cp, shrink = 0.25)");
	PyRun_StringStd("ax.view_init(27, -21)");

	PyRun_StringStd("ax.text2D(0.03, 0.5, 'p(Y)', transform = ax.transAxes)");
	
	PyRun_StringStd("filename = './data/Distribution'");
	PyRun_StringStd("np.savez_compressed(filename, a = hx, b = hy, c = hz, d = " + std::to_string(Sine) + ")");
	
	//PyRun_StringStd("mxx = np.mean(csetx.allsegs[30][0][:, 1])");
	//PyRun_StringStd("mxy = np.mean(csety.allsegs[30][0][:, 1])");

	//PyRun_StringStd("print(mxx)");
	//std::cout << std::endl;
	//PyRun_StringStd("print(mxy)");
	//std::cout << std::endl;

}

void plot_matplotlib::line(const double x1, const double x2, const double y1, const double y2,
	const std::string color, const double linewidth, const std::string linestyle)
{
	PyRun_StringStd("plt.plot([" + std::to_string(x1) + ", "
		+ std::to_string(x2) + "], [" + std::to_string(y1) + ", " + std::to_string(y2) + "], color = '" + color + "',\
		linewidth= " + std::to_string(linewidth) + ", linestyle =\
 '" + linestyle + "')");
}

void plot_matplotlib::text(const double x, const double y,
	std::string somestring, std::string color, double fontsize)
{
	PyRun_StringStd("plt.text(" + std::to_string(x) + ", " + std::to_string(y) + ", '" + somestring + "', color = \
      '" + color + "', fontsize = " + std::to_string(fontsize) + ")");
}

void plot_matplotlib::arrow(const double x_tail, const double y_tail,
	const double x_head, const double y_head, std::string color)
{
	//PyRun_StringStd("plt.arrow(" + to_string(x) + ", " + to_string(y) + "," + to_string(dx) + ", \
    //" + to_string(dy) + ", head_width=1, linewidth = 5, facecolor = '" + color + "')");

	PyRun_StringStd("import matplotlib.patches as mpatches");
	PyRun_StringStd("x_tail = " + std::to_string(x_tail) + "");//3.4
	PyRun_StringStd("y_tail = " + std::to_string(y_tail) + "");//2020
	PyRun_StringStd("x_head = " + std::to_string(x_head) + "");//2.1
	PyRun_StringStd("y_head = " + std::to_string(y_head) + "");//2020
	PyRun_StringStd("dx = x_head - x_tail");
	PyRun_StringStd("dy = y_head - y_tail");
	PyRun_StringStd("arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head), \
    color = '" + color + "', mutation_scale = 10)");
	PyRun_StringStd("ax = plt.gca()");
	PyRun_StringStd("ax.add_patch(arrow)");
}

void plot_matplotlib::init_plot_window(const char* name, int x, int y)
{
	run_customcommand(name);
	// PyRun_Simple("fig = plt.figure(1)");
	PyRun_StringStd("wm = plt.get_current_fig_manager()");
	PyRun_StringStd("wm.window.wm_geometry('+" + std::to_string(x) + "+" + std::to_string(y) + "')");
	adjust_ticker();
	PyRun_StringStd("import atexit");
	PyRun_StringStd("import sys");
	PyRun_StringStd("import subprocess");
}

std::vector<std::string> plot_matplotlib::colors
{
	/*
			"cloudy blue",
	"dark pastel green",
	"dust",
	"electric lime",
	"fresh green",
	"light eggplant",
	"nasty green",
	"really light blue",
	"tea",
	"warm purple",
	"yellowish tan",
	"cement",
	"dark grass green",
	"dusty teal",
	"grey teal",
	"macaroni and cheese",
	"pinkish tan",
	"spruce",
	"strong blue",
	"toxic green",
	"windows blue",
	"blue blue",
	"blue with a hint of purple",
	"booger",
	"bright sea green",
	"dark green blue",
	"deep turquoise",
	"green teal",

	 */ "strong pink",
	"bland",
	"deep aqua",
	"lavender pink",
	"light moss green",
	"light seafoam green",
	"olive yellow",
	"pig pink",
	"deep lilac",
	"desert",
	"dusty lavender",
	"purpley grey",
	"purply",
	"candy pink",
	"light pastel green",
	"boring green",
	"kiwi green",
	"light grey green",
	"orange pink",
	"tea green",
	"very light brown",
	"egg shell",
	"eggplant purple",
	"powder pink",
	"reddish grey",
	"baby shit brown",
	"liliac",
	"stormy blue",
	"ugly brown",
	"custard",
	"darkish pink",
	"deep brown",
	"greenish beige",
	"manilla",
	"off blue",
	"battleship grey",
	"browny green",
	"bruise",
	"kelley green",
	"sickly yellow",
	"sunny yellow",
	"azul",
	"darkgreen",
	"green/yellow",
	"lichen",
	"light light green",
	"pale gold",
	"sun yellow",
	"tan green",
	"burple",
	"butterscotch",
	"toupe",
	"dark cream",
	"indian red",
	"light lavendar",
	"poison green",
	"baby puke green",
	"bright yellow green",
	"charcoal grey",
	"squash",
	"cinnamon",
	"light pea green",
	"radioactive green",
	"raw sienna",
	"baby purple",
	"cocoa",
	"light royal blue",
	"orangeish",
	"rust brown",
	"sand brown",
	"swamp",
	"tealish green",
	"burnt siena",
	"camo",
	"dusk blue",
	"fern",
	"old rose",
	"pale light green",
	"peachy pink",
	"rosy pink",
	"light bluish green",
	"light bright green",
	"light neon green",
	"light seafoam",
	"tiffany blue",
	"washed out green",
	"browny orange",
	"nice blue",
	"sapphire",
	"greyish teal",
	"orangey yellow",
	"parchment",
	"straw",
	"very dark brown",
	"terracota",
	"ugly blue",
	"clear blue",
	"creme",
	"foam green",
	"grey/green",
	"light gold",
	"seafoam blue",
	"topaz",
	"violet pink",
	"wintergreen",
	"yellow tan",
	"dark fuchsia",
	"indigo blue",
	"light yellowish green",
	"pale magenta",
	"rich purple",
	"sunflower yellow",
	"green/blue",
	"leather",
	"racing green",
	"vivid purple",
	"dark royal blue",
	"hazel",
	"muted pink",
	"booger green",
	"canary",
	"cool grey",
	"dark taupe",
	"darkish purple",
	"true green",
	"coral pink",
	"dark sage",
	"dark slate blue",
	"flat blue",
	"mushroom",
	"rich blue",
	"dirty purple",
	"greenblue",
	"icky green",
	"light khaki",
	"warm blue",
	"dark hot pink",
	"deep sea blue",
	"carmine",
	"dark yellow green",
	"pale peach",
	"plum purple",
	"golden rod",
	"neon red",
	"old pink",
	"very pale blue",
	"blood orange",
	"grapefruit",
	"sand yellow",
	"clay brown",
	"dark blue grey",
	"flat green",
	"light green blue",
	"warm pink",
	"dodger blue",
	"gross green",
	"ice",
	"metallic blue",
	"pale salmon",
	"sap green",
	"algae",
	"bluey grey",
	"greeny grey",
	"highlighter green",
	"light light blue",
	"light mint",
	"raw umber",
	"vivid blue",
	"deep lavender",
	"dull teal",
	"light greenish blue",
	"mud green",
	"pinky",
	"red wine",
	"shit green",
	"tan brown",
	"darkblue",
	"rosa",
	"lipstick",
	"pale mauve",
	"claret",
	"dandelion",
	"orangered",
	"poop green",
	"ruby",
	"dark",
	"greenish turquoise",
	"pastel red",
	"piss yellow",
	"bright cyan",
	"dark coral",
	"algae green",
	"darkish red",
	"reddy brown",
	"blush pink",
	"camouflage green",
	"lawn green",
	"putty",
	"vibrant blue",
	"dark sand",
	"purple/blue",
	"saffron",
	"twilight",
	"warm brown",
	"bluegrey",
	"bubble gum pink",
	"duck egg blue",
	"greenish cyan",
	"petrol",
	"royal",
	"butter",
	"dusty orange",
	"off yellow",
	"pale olive green",
	"orangish",
	"leaf",
	"light blue grey",
	"dried blood",
	"lightish purple",
	"rusty red",
	"lavender blue",
	"light grass green",
	"light mint green",
	"sunflower",
	"velvet",
	"brick orange",
	"lightish red",
	"pure blue",
	"twilight blue",
	"violet red",
	"yellowy brown",
	"carnation",
	"muddy yellow",
	"dark seafoam green",
	"deep rose",
	"dusty red",
	"grey/blue",
	"lemon lime",
	"purple/pink",
	"brown yellow",
	"purple brown",
	"wisteria",
	"banana yellow",
	"lipstick red",
	"water blue",
	"brown grey",
	"vibrant purple",
	"baby green",
	"barf green",
	"eggshell blue",
	"sandy yellow",
	"cool green",
	"pale",
	"blue/grey",
	"hot magenta",
	"greyblue",
	"purpley",
	"baby shit green",
	"brownish pink",
	"dark aquamarine",
	"diarrhea",
	"light mustard",
	"pale sky blue",
	"turtle green",
	"bright olive",
	"dark grey blue",
	"greeny brown",
	"lemon green",
	"light periwinkle",
	"seaweed green",
	"sunshine yellow",
	"ugly purple",
	"medium pink",
	"puke brown",
	"very light pink",
	"viridian",
	"bile",
	"faded yellow",
	"very pale green",
	"vibrant green",
	"bright lime",
	"spearmint",
	"light aquamarine",
	"light sage",
	"yellowgreen",
	"baby poo",
	"dark seafoam",
	"deep teal",
	"heather",
	"rust orange",
	"dirty blue",
	"fern green",
	"bright lilac",
	"weird green",
	"peacock blue",
	"avocado green",
	"faded orange",
	"grape purple",
	"hot green",
	"lime yellow",
	"mango",
	"shamrock",
	"bubblegum",
	"purplish brown",
	"vomit yellow",
	"pale cyan",
	"key lime",
	"tomato red",
	"lightgreen",
	"merlot",
	"night blue",
	"purpleish pink",
	"apple",
	"baby poop green",
	"green apple",
	"heliotrope",
	"yellow/green",
	"almost black",
	"cool blue",
	"leafy green",
	"mustard brown",
	"dusk",
	"dull brown",
	"frog green",
	"vivid green",
	"bright light green",
	"fluro green",
	"kiwi",
	"seaweed",
	"navy green",
	"ultramarine blue",
	"iris",
	"pastel orange",
	"yellowish orange",
	"perrywinkle",
	"tealish",
	"dark plum",
	"pear",
	"pinkish orange",
	"midnight purple",
	"light urple",
	"dark mint",
	"greenish tan",
	"light burgundy",
	"turquoise blue",
	"ugly pink",
	"sandy",
	"electric pink",
	"muted purple",
	"mid green",
	"greyish",
	"neon yellow",
	"banana",
	"carnation pink",
	"tomato",
	"sea",
	"muddy brown",
	"turquoise green",
	"buff",
	"fawn",
	"muted blue",
	"pale rose",
	"dark mint green",
	"amethyst",
	"blue/green",
	"chestnut",
	"sick green",
	"pea",
	"rusty orange",
	"stone",
	"rose red",
	"pale aqua",
	"deep orange",
	"earth",
	"mossy green",
	"grassy green",
	"pale lime green",
	"light grey blue",
	"pale grey",
	"asparagus",
	"blueberry",
	"purple red",
	"pale lime",
	"greenish teal",
	"caramel",
	"deep magenta",
	"light peach",
	"milk chocolate",
	"ocher",
	"off green",
	"purply pink",
	"lightblue",
	"dusky blue",
	"golden",
	"light beige",
	"butter yellow",
	"dusky purple",
	"french blue",
	"ugly yellow",
	"greeny yellow",
	"orangish red",
	"shamrock green",
	"orangish brown",
	"tree green",
	"deep violet",
	"gunmetal",
	"blue/purple",
	"cherry",
	"sandy brown",
	"warm grey",
	"dark indigo",
	"midnight",
	"bluey green",
	"grey pink",
	"soft purple",
	"blood",
	"brown red",
	"medium grey",
	"berry",
	"poo",
	"purpley pink",
	"light salmon",
	"snot",
	"easter purple",
	"light yellow green",
	"dark navy blue",
	"drab",
	"light rose",
	"rouge",
	"purplish red",
	"slime green",
	"baby poop",
	"irish green",
	"pink/purple",
	"dark navy",
	"greeny blue",
	"light plum",
	"pinkish grey",
	"dirty orange",
	"rust red",
	"pale lilac",
	"orangey red",
	"primary blue",
	"kermit green",
	"brownish purple",
	"murky green",
	"wheat",
	"very dark purple",
	"bottle green",
	"watermelon",
	"deep sky blue",
	"fire engine red",
	"yellow ochre",
	"pumpkin orange",
	"pale olive",
	"light lilac",
	"lightish green",
	"carolina blue",
	"mulberry",
	"shocking pink",
	"auburn",
	"bright lime green",
	"celadon",
	"pinkish brown",
	"poo brown",
	"bright sky blue",
	"celery",
	"dirt brown",
	"strawberry",
	"dark lime",
	"copper",
	"medium brown",
	"muted green",
	"robin's egg",
	"bright aqua",
	"bright lavender",
	"ivory",
	"very light purple",
	"light navy",
	"pink red",
	"olive brown",
	"poop brown",
	"mustard green",
	"ocean green",
	"very dark blue",
	"dusty green",
	"light navy blue",
	"minty green",
	"adobe",
	"barney",
	"jade green",
	"bright light blue",
	"light lime",
	"dark khaki",
	"orange yellow",
	"ocre",
	"maize",
	"faded pink",
	"british racing green",
	"sandstone",
	"mud brown",
	"light sea green",
	"robin egg blue",
	"aqua marine",
	"dark sea green",
	"soft pink",
	"orangey brown",
	"cherry red",
	"burnt yellow",
	"brownish grey",
	"camel",
	"purplish grey",
	"marine",
	"greyish pink",
	"pale turquoise",
	"pastel yellow",
	"bluey purple",
	"canary yellow",
	"faded red",
	"sepia",
	"coffee",
	"bright magenta",
	"mocha",
	"ecru",
	"purpleish",
	"cranberry",
	"darkish green",
	"brown orange",
	"dusky rose",
	"melon",
	"sickly green",
	"silver",
	"purply blue",
	"purpleish blue",
	"hospital green",
	"shit brown",
	"mid blue",
	"amber",
	"easter green",
	"soft blue",
	"cerulean blue",
	"golden brown",
	"bright turquoise",
	"red pink",
	"red purple",
	"greyish brown",
	"vermillion",
	"russet",
	"steel grey",
	"lighter purple",
	"bright violet",
	"prussian blue",
	"slate green",
	"dirty pink",
	"dark blue green",
	"pine",
	"yellowy green",
	"dark gold",
	"bluish",
	"darkish blue",
	"dull red",
	"pinky red",
	"bronze",
	"pale teal",
	"military green",
	"barbie pink",
	"bubblegum pink",
	"pea soup green",
	"dark mustard",
	"shit",
	"medium purple",
	"very dark green",
	"dirt",
	"dusky pink",
	"red violet",
	"lemon yellow",
	"pistachio",
	"dull yellow",
	"dark lime green",
	"denim blue",
	"teal blue",
	"lightish blue",
	"purpley blue",
	"light indigo",
	"swamp green",
	"brown green",
	"dark maroon",
	"hot purple",
	"dark forest green",
	"faded blue",
	"drab green",
	"light lime green",
	"snot green",
	"yellowish",
	"light blue green",
	"bordeaux",
	"light mauve",
	"ocean",
	"marigold",
	"muddy green",
	"dull orange",
	"steel",
	"electric purple",
	"fluorescent green",
	"yellowish brown",
	"blush",
	"soft green",
	"bright orange",
	"lemon",
	"purple grey",
	"acid green",
	"pale lavender",
	"violet blue",
	"light forest green",
	"burnt red",
	"khaki green",
	"cerise",
	"faded purple",
	"apricot",
	"dark olive green",
	"grey brown",
	"green grey",
	"true blue",
	"pale violet",
	"periwinkle blue",
	"light sky blue",
	"blurple",
	"green brown",
	"bluegreen",
	"bright teal",
	"brownish yellow",
	"pea soup",
	"forest",
	"barney purple",
	"ultramarine",
	"purplish",
	"puke yellow",
	"bluish grey",
	"dark periwinkle",
	"dark lilac",
	"reddish",
	"light maroon",
	"dusty purple",
	"terra cotta",
	"avocado",
	"marine blue",
	"teal green",
	"slate grey",
	"lighter green",
	"electric green",
	"dusty blue",
	"golden yellow",
	"bright yellow",
	"light lavender",
	"umber",
	"poop",
	"dark peach",
	"jungle green",
	"eggshell",
	"denim",
	"yellow brown",
	"dull purple",
	"chocolate brown",
	"wine red",
	"neon blue",
	"dirty green",
	"light tan",
	"ice blue",
	"cadet blue",
	"dark mauve",
	"very light blue",
	"grey purple",
	"pastel pink",
	"very light green",
	"dark sky blue",
	"evergreen",
	"dull pink",
	"aubergine",
	"mahogany",
	"reddish orange",
	"deep green",
	"vomit green",
	"purple pink",
	"dusty pink",
	"faded green",
	"camo green",
	"pinky purple",
	"pink purple",
	"brownish red",
	"dark rose",
	"mud",
	"brownish",
	"emerald green",
	"pale brown",
	"dull blue",
	"burnt umber",
	"medium green",
	"clay",
	"light aqua",
	"light olive green",
	"brownish orange",
	"dark aqua",
	"purplish pink",
	"dark salmon",
	"greenish grey",
	"jade",
	"ugly green",
	"dark beige",
	"emerald",
	"pale red",
	"light magenta",
	"sky",
	"light cyan",
	"yellow orange",
	"reddish purple",
	"reddish pink",
	"orchid",
	"dirty yellow",
	"orange red",
	"deep red",
	"orange brown",
	"cobalt blue",
	"neon pink",
	"rose pink",
	"greyish purple",
	"raspberry",
	"aqua green",
	"salmon pink",
	"tangerine",
	"brownish green",
	"red brown",
	"greenish brown",
	"pumpkin",
	"pine green",
	"charcoal",
	"baby pink",
	"cornflower",
	"blue violet",
	"chocolate",
	"greyish green",
	"scarlet",
	"green yellow",
	"dark olive",
	"sienna",
	"pastel purple",
	"terracotta",
	"aqua blue",
	"sage green",
	"blood red",
	"deep pink",
	"grass",
	"moss",
	"pastel blue",
	"bluish green",
	"green blue",
	"dark tan",
	"greenish blue",
	"pale orange",
	"vomit",
	"forrest green",
	"dark lavender",
	"dark violet",
	"purple blue",
	"dark cyan",
	"olive drab",
	"pinkish",
	"cobalt",
	"neon purple",
	"light turquoise",
	"apple green",
	"dull green",
	"wine",
	"powder blue",
	"off white",
	"electric blue",
	"dark turquoise",
	"blue purple",
	"azure",
	"bright red",
	"pinkish red",
	"cornflower blue",
	"light olive",
	"grape",
	"greyish blue",
	"purplish blue",
	"yellowish green",
	"greenish yellow",
	"medium blue",
	"dusty rose",
	"light violet",
	"midnight blue",
	"bluish purple",
	"red orange",
	"dark magenta",
	"greenish",
	"ocean blue",
	"coral",
	"cream",
	"reddish brown",
	"burnt sienna",
	"brick",
	"sage",
	"grey green",
	"white",
	"robin's egg blue",
	"moss green",
	"steel blue",
	"eggplant",
	"light yellow",
	"leaf green",
	"light grey",
	"puke",
	"pinkish purple",
	"sea blue",
	"pale purple",
	"slate blue",
	"blue grey",
	"hunter green",
	"fuchsia",
	"crimson",
	"pale yellow",
	"ochre",
	"mustard yellow",
	"light red",
	"cerulean",
	"pale pink",
	"deep blue",
	"rust",
	"light teal",
	"slate",
	"goldenrod",
	"dark yellow",
	"dark grey",
	"army green",
	"grey blue",
	"seafoam",
	"puce",
	"spring green",
	"dark orange",
	"sand",
	"pastel green",
	"mint",
	"light orange",
	"bright pink",
	"chartreuse",
	"deep purple",
	"dark brown",
	"taupe",
	"pea green",
	"puke green",
	"kelly green",
	"seafoam green",
	"blue green",
	"khaki",
	"burgundy",
	"dark teal",
	"brick red",
	"royal purple",
	"plum",
	"mint green",
	"gold",
	"baby blue",
	"yellow green",
	"bright purple",
	"dark red",
	"pale blue",
	"grass green",
	"navy",
	"aquamarine",
	"burnt orange",
	"neon green",
	"bright blue",
	"rose",
	"light pink",
	"mustard",
	"indigo",
	"lime",
	"sea green",
	"periwinkle",
	"dark pink",
	"olive green",
	"peach",
	"pale green",
	"light brown",
	"hot pink",
	"black",
	"lilac",
	"navy blue",
	"royal blue",
	"beige",
	"salmon",
	"olive",
	"maroon",
	"bright green",
	"dark purple",
	"mauve",
	"forest green",
	"aqua",
	"cyan",
	"tan",
	"dark blue",
	"lavender",
	"turquoise",
	"dark green",
	"violet",
	"light purple",
	"lime green",
	"grey",
	"sky blue",
	"yellow",
	"magenta",
	"light green",
	"orange",
	"teal",
	"light blue",
	"red",
	"brown",
	"pink",
	"blue",
	"green",
	"purple" };
