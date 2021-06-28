#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "clipper/clipper.hpp"
#include "ThreadPool/ThreadPool.h"

#include <iostream>
#include <fstream>
#include <queue>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

using namespace std;
using namespace cv;

#define THREAD_NUMS 8

namespace py = pybind11;

namespace post_process{

    double get_perimeter(const vector<Point> &poly)
    {
        double peri = 0.0;
        int poly_size = poly.size();
        for(int i = 0; i < poly_size - 1; i++)
        {
            Point point_0 = poly[i];
            Point point_1 = poly[i + 1];
            double dist_tmp = sqrt(pow(point_0.x - point_1.x, 2) + pow(point_0.y - point_1.y, 2));
            peri += dist_tmp;
        }
        return peri;
    }

    void get_Mats(const uchar *kernel_data, vector<long int> kernel_shape, Mat &kernel,
                  const float *score_data, vector<long int> score_shape, Mat &score)
    {
        for (int x = 0; x < kernel.rows; ++x)
        {
            for(int y = 0; y < kernel.cols; ++y)
            {
                kernel.at<uchar>(x, y) = kernel_data[x * kernel_shape[1] + y] * 255;
                score.at<float>(x, y) = score_data[x * kernel_shape[1] + y];
            }
        }
    }

    vector<int> get_bbox(vector<int>& poly, int score_h, int score_w)
    {
        int xmin = score_w - 1, ymin = score_h - 1, xmax = 0, ymax = 0;
        int point_num = poly.size() / 2;
        for(int i = 0; i < point_num; i++)
        {
            xmin = min(xmin, poly[2 * i]);
            xmax = max(xmax, poly[2 * i]);
            ymin = min(ymin, poly[2 * i + 1]);
            ymax = max(ymax, poly[2 * i + 1]);
        }
        vector<int> res{xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax};
        return res;
    }

    vector<int> get_bbox(const vector<Point>& poly, int score_h, int score_w)
    {
        int xmin = score_w - 1, ymin = score_h - 1, xmax = 0, ymax = 0;
        int point_num = poly.size();
        for(int i = 0; i < point_num; i++)
        {
            xmin = min(xmin, poly[i].x);
            xmax = max(xmax, poly[i].x);
            ymin = min(ymin, poly[i].y);
            ymax = max(ymax, poly[i].y);
        }
        vector<int> res{xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax};
        return res;
    }

    vector<int> get_dilated_poly(const vector<Point>& poly,
                                 Mat score,
                                 int score_h, int score_w,
                                 int min_border,
                                 int min_area_thr,
                                 float score_thr,
                                 float expand_scale)
    {
        vector<int> dilated_poly;
        if (poly.size() <= 2){ return dilated_poly; }

        // check min_border
        vector<int> poly_bbox = get_bbox(poly, score_h, score_w);
        int poly_xmin = poly_bbox[0], poly_ymin = poly_bbox[1];
        int poly_h = poly_bbox[5] - poly_bbox[1] + 1;
        int poly_w = poly_bbox[4] - poly_bbox[0] + 1;
        if(min(poly_h, poly_w) < min_border) { return dilated_poly; }

        Mat mask = Mat::zeros(poly_h + 4, poly_w + 4, CV_8UC1);

        vector<Point> _poly;
        vector<vector<Point> > poly_arr;
        for (const auto& point: poly)
        {
            Point _point;
            _point.x = point.x - poly_xmin + 2;
            _point.y = point.y - poly_ymin + 2;
            _poly.push_back(_point);
        }
        poly_arr.push_back(_poly);
        drawContours(mask, poly_arr, 0, 255, -1);

        // check the score
        uchar *p;
        int count = 0;
        float avg_score = 0.0;
        for (int i = 2; i < 2 + poly_h; i++)
        {
            p = mask.ptr<uchar>(i);
            for (int j = 2; j < 2 + poly_w; j++)
            {
                if (p[j] == 255)
                {
                    count += 1;
                    avg_score += score.ptr<float>(i - 2 + poly_ymin)[j - 2 + poly_xmin];
                }
            }
        }
        mask.release();

        if (count < min_area_thr) { return dilated_poly; }
        else { avg_score /= count; }
        if (avg_score < score_thr) { return dilated_poly; }

        double poly_peri = 0.0;
        double area = 0.0;

        ClipperLib::Path subj;
        ClipperLib::Paths solution;
        for (auto &point : poly){ subj << ClipperLib::IntPoint(point.x, point.y); }

        area = abs(ClipperLib::Area(subj));
        poly_peri = get_perimeter(poly);
        //poly_peri = arcLength(approx, true);
        int offset = min(int(area * expand_scale / poly_peri), 20);

        ClipperLib::ClipperOffset co;
        co.AddPath(subj, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
        co.Execute(solution, offset);

        if (solution.size() == 0 || solution.size() > 1) { return dilated_poly; }

        for (auto &point : solution[0])
        {
            dilated_poly.push_back(point.X);
            dilated_poly.push_back(point.Y);
        }

        return dilated_poly;
    }

    vector<vector<int>> simple_dilate(py::array_t<uchar, py::array::c_style | py::array::forcecast> kernel_py,
                                      py::array_t<float, py::array::c_style | py::array::forcecast> score_py,
                                      float score_thr=0.95,
                                      float expand_scale=1.8,
                                      int min_area_thr=32,
                                      int min_border=3)
    {
        vector<vector<int> > dilated_polys;

        auto kernel_buf = kernel_py.request();
        auto kernel_data = static_cast<uchar *>(kernel_buf.ptr);
        auto score_buf = score_py.request();
        auto score_data = static_cast<float *>(score_buf.ptr);

        Mat kernel = Mat::zeros(kernel_buf.shape[0], kernel_buf.shape[1], CV_8UC1);
        Mat score = Mat::zeros(score_buf.shape[0], score_buf.shape[1], CV_32FC1);
        int score_h = score.rows;
        int score_w = score.cols;
        get_Mats(kernel_data, kernel_buf.shape, kernel, score_data, score_buf.shape, score);

        vector<vector<Point> > polys;
        vector<Vec4i> hierarchy;
        findContours(kernel, polys, hierarchy,
                CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        ThreadPool pool(THREAD_NUMS);
        std::vector< std::future<std::vector<int>> > results;

        for(const auto &poly : polys)
        {
            results.emplace_back(
                pool.enqueue(get_dilated_poly,
                             poly, score, score_h, score_w, min_border,
                             min_area_thr, score_thr, expand_scale)
            );
        }

        for(auto && result: results)
        {
            auto dilated_poly = result.get();
            if (dilated_poly.size() > 0)
                dilated_polys.push_back(dilated_poly);
        }

        return dilated_polys;
    }
}


PYBIND11_PLUGIN(simple_dilate) {
    py::module m("simple_dilate", "a simple way to replace PSE.");
    m.def("simple_dilate", &post_process::simple_dilate,
          "decode the model's prediction to get the text polygons.");
    return m.ptr();
}



