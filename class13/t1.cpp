#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main()
{
	Mat refMat,img, tempMat, result, frame;
	VideoCapture capVideo(0);
	int cnt = 0;
	while (1) {
		capVideo >> frame;
		if (cnt == 0) {
			Rect2d r;
			r = selectROI(frame, true);
			tempMat = frame(r);
			tempMat.copyTo(refMat);
			destroyAllWindows();
		}
		int result_cols = frame.cols - tempMat.cols + 1;
		int result_rows = frame.rows - tempMat.rows + 1;
		result.create(result_cols, result_rows, CV_32FC1);

		matchTemplate(frame, tempMat, result, TM_SQDIFF_NORMED);
		normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

		double minVal;
		double maxVal;
		Point minLoc;
		Point maxLoc;
		Point matchLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

		matchLoc = maxLoc;

		rectangle(frame, Point(matchLoc.x, matchLoc.y - 0.3 * tempMat.rows), Point(matchLoc.x + tempMat.cols, matchLoc.y + 0.7*tempMat.rows), Scalar(0, 255, 0), 2, 8, 0);

		imshow("frame", frame);
		waitKey(3);
		cnt++;
	}
	}