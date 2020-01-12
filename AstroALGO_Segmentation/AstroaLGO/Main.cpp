#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#define __validate255(x) x = x > 255 ? 255 : x;
#define __validate0(x) x = x < 0 ? 0 : x;

struct Point
{
	int x, y;
};

struct Rect
{
	Point stLeftTop;
	Point stRightBottom;
};

struct Rect4
{
	int top{};
	int bottom{};
	int left{};
	int right{};

	bool PtOnRect(int x, int y) {
		return x >= left && x <= right && y >= top && y <= bottom;
	}

	bool PtInRect(int x, int y) {
		return false;
	}

};

struct Elipse
{
	int cx{};
	int cy{};
	int rx{};
	int ry{};

	Elipse(const Rect4& rect) {
		rx = (rect.right - rect.left) / 2;
		ry = (rect.bottom - rect.top) / 2;
		cx = rect.left + rx;
		cy = rect.top + ry;
	}

	bool PtInElipse(int x, int y) {
		return (((x - cx) * (x - cx)) / (rx * rx)) + (((y - cy) * (y - cy)) / (ry * ry)) <= 1;
	}

};

namespace {
	const std::string SrcPath1 = "fbs0005-DFBSJ142114.64+425823.6.jpg";
	const std::string SrcPath2 = "fbs0005-DFBSJ142122.01+425921.3.jpg";
	const std::string SrcPath3 = "fbs0007-DFBSJ150111.23+425941.7.jpg";
	const std::string SrcPath4 = "fbs0785M-DFBSJ071746.00+470631.8.jpg";
	const std::string SrcPathOut1 = "fbs0005-DFBSJ142114.64+425823.6_out.jpg";
	const std::string SrcPathOut2 = "fbs0005-DFBSJ142122.01+425921.3_out.jpg";
	const std::string SrcPathOut3 = "fbs0007-DFBSJ150111.23+425941.7_out.jpg";
	const std::string SrcPathOut4 = "fbs0785M-DFBSJ071746.00+470631.8_out.jpg";
}

void Segmentation(cv::Mat& src, const int t, const int iRowCount) {

	std::vector<int> hist(256, 0);

	int iWidth = src.cols;
	int iHeight = src.rows;

	//int max = 0;
	//int min = std::numeric_limits<int>::max();

	for (int y = 0; y < iHeight; y++)
	{
		for (int x = 0; x < iWidth; x++)
		{
			auto pixel = src.at<cv::Vec3b>(y, x);
			int result = (pixel[0] + pixel[1] + pixel[2]) / 3;
			src.at<cv::Vec3b>(y, x)[0] = result;
			src.at<cv::Vec3b>(y, x)[1] = result;
			src.at<cv::Vec3b>(y, x)[2] = result;
			hist[result] += 1;
			//max = std::max(max, result);
			//min = std::min(min, result);
		}
	}

	/*std::cout << max << std::endl;
	std::cout << min << std::endl;

	for (int y = 0; y < iHeight; y++)
	{
		for (int x = 0; x < iWidth; x++)
		{
			auto pixelVal = src.at<cv::Vec3b>(y, x)[0];
			if (pixelVal > t) {
				src.at<cv::Vec3b>(y, x)[0] = 255;
				src.at<cv::Vec3b>(y, x)[1] = 255;
				src.at<cv::Vec3b>(y, x)[2] = 255;
			}
			else {
				src.at<cv::Vec3b>(y, x)[0] = 0;
				src.at<cv::Vec3b>(y, x)[1] = 0;
				src.at<cv::Vec3b>(y, x)[2] = 0;
			}
		}
	}*/


	int avg = 0;
	int count = 0;
	for (int i = 1; i < hist.size(); i++)
	{
		avg += i * hist[i];
		count += hist[i];
	}

	int th = avg / count;
	th += th << 1;
	__validate255(th);


	for (int y = 0; y < iHeight; y++)
	{
		for (int x = 0; x < iWidth; x++)
		{
			auto pixelVal = src.at<cv::Vec3b>(y, x)[0];
			if (pixelVal > th) {
				src.at<cv::Vec3b>(y, x)[0] = 255;
				src.at<cv::Vec3b>(y, x)[1] = 255;
				src.at<cv::Vec3b>(y, x)[2] = 255;
			}
			else {
				src.at<cv::Vec3b>(y, x)[0] = 0;
				src.at<cv::Vec3b>(y, x)[1] = 0;
				src.at<cv::Vec3b>(y, x)[2] = 0;
			}
		}
	}

}

void Erosion(cv::Mat& src, int iSize) {
	int iWidth = src.cols;
	int iHeight = src.rows;

	for (int x = 0; x < iWidth; x++)
	{
		bool segment = false;
		int iBegin = 0;
		int iEnd = 0;
		for (int y = 0; y < iHeight; y++)
		{
			auto pixel = src.at<uchar>(y, x);
			if (pixel == 255) {
				if(segment) {
					int d = iEnd - iBegin;
					if (d <= iSize) {
						for (int dy = iBegin; dy <= iEnd; dy++)
						{
							src.at<uchar>(dy, x) = 255;
						}
					}
					segment = false;
				}
			}
			else {
				if (!segment) {
					iBegin = y;
					segment = true;
				}
				iEnd = y;
			}
		}
	}

}

Rect4 RectSegmentation(cv::Mat& src) {
	
	Rect4 result;
	int iWidth = src.cols;
	int iHeight = src.rows;

	int cX = (iWidth + 1) >> 1;
	int cY = (iHeight + 1) >> 1;

	std::vector<bool> lines;
	
	auto pixel = src.at<uchar>(cY, cX);
	if (pixel == 0) {
		bool iterTop = true;
		bool iterBottom = true;
		int yTop = cY;
		int yBottom = cY;
		while (iterTop || iterBottom)
		{
			auto topPixel = src.at<uchar>(yTop, cX);
			auto bottomPixel = src.at<uchar>(yBottom, cX);

			if (topPixel == 0 && iterTop) {
				--yTop;
			}
			else {
				iterTop = false;
			}

			if (bottomPixel == 0 && iterBottom) {
				++yBottom;
			}
			else {
				iterBottom = false;
			}

		}

		result.top = yTop - 5 >= 0 ? yTop - 5 : yTop;
		result.bottom = yBottom + 10 < iHeight ? yBottom + 10 : yBottom;

		bool iterLeft = true;
		bool iterRight = true;
		int xLeft = cX;
		int xRight = cX;

		while (iterLeft || iterRight)
		{
			int countXLeft = 0;
			int countXRight = 0;
			for (int y = result.top; y <= result.bottom; y++)
			{
				
				auto pixelLeft = src.at<uchar>(y, xLeft);
				auto pixelRight = src.at<uchar>(y, xRight);

				if (pixelLeft == 0) {
					++countXLeft;
				}
				if (pixelRight == 0) {
					++countXRight;
				}
			}

			if (countXLeft > 0 && iterLeft) {
				--xLeft;
			}
			else {
				iterLeft = false;
			}

			if (countXRight > 0 && iterRight) {
				++xRight;
			}
			else {
				iterRight = false;
			}

		}

		result.left = xLeft - 2 >= 0 ? xLeft - 2 : xLeft;
		result.right = xRight + 2 < iWidth ? xRight + 2 : xRight;

	}
	else {
		//TODO
	}
	
	
	return result;

}

void DrawRectContour(cv::Mat& src, Rect4 rect) 
{
	int iWidth = src.cols;
	int iHeight = src.rows;

	for (int y = 0; y < iHeight; y++)
	{
		for (int x = 0; x < iWidth; x++)
		{
			if (!rect.PtOnRect(x, y)) {
				src.at<uchar>(y, x) = 0;
			}
		}
	}

}

int main() {


	cv::Mat img = cv::imread(SrcPath1);
	cv::Mat gray = img.clone();
	cv::Mat grayResult = img.clone();
	std::cout << img.cols << std::endl;
	std::cout << img.rows << std::endl;

	cv::Mat blured = img.clone();
	cv::Mat resultImg = img.clone();
	cv::GaussianBlur(img, blured, cv::Size(3, 3), 0, 0);

	cv::cvtColor(blured, gray, cv::COLOR_RGBA2GRAY, 0);
	cv::cvtColor(img, grayResult, cv::COLOR_RGBA2GRAY, 0);
	cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

	//Segmentation(blured, 128, 10);

	//DrawContours(bluredCopy, blured);

	Erosion(gray, 10);
	Rect4 result = RectSegmentation(gray);
	DrawRectContour(grayResult, result);


	//imwrite(SrcPathOut1, grayResult);
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	cv::imshow("image", grayResult);
	cv::waitKey(0);

	return 0;
}