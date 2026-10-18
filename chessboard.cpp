#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include<string>

using namespace cv;
using namespace std;

int main()
{
	float squareSize = 50.0f;
	vector<vector<Point3f>>allobjectPoints;
	vector<vector<Point2f>>allimgPoints;
	Size chessboard(11, 8);
	vector<Point3f> objectsPoints;
	for (int row = 0;row < chessboard.height;row++)
	{
		for (int col = 0;col < chessboard.width;col++)
		{
			objectsPoints.push_back(Point3f(col * squareSize, row * squareSize, 0));
		}
	}
	Mat gray;
	vector<String> imagePaths;
	glob("C:/calibration_images/calibration_images/*.png", imagePaths);
	for (int i = 0;i < imagePaths.size();i++)
	{
		Mat img = imread(imagePaths[i]);
		Size imgSize = img.size();
		cvtColor(img, gray, COLOR_BGR2GRAY);

		vector<Point2f> corners;
		bool found = findChessboardCorners(gray, chessboard, corners);
		if (found)
		{
			cornerSubPix(gray, corners, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.01));
			drawChessboardCorners(img, chessboard, corners, found);
			vector<vector<Point3f>> allobjectPoints;
			vector<vector<Point2f>> allimagePoints;
			allobjectPoints.push_back(objectsPoints);
			allimagePoints.push_back(corners);
			Mat CameraMatrix, diffcoeffs, rvecs, tvecs;
			double rms = calibrateCamera(allobjectPoints, allimagePoints,imgSize , CameraMatrix, diffcoeffs, rvecs, tvecs);
			FileStorage fs("C:/cameracaliberate/camera.yml", FileStorage::WRITE);
			fs << "CameraMatrix" << CameraMatrix;
			fs << "diffcoeffs" << diffcoeffs;
			fs << "width" << imgSize.width;
			fs << "height" << imgSize.height;
			fs.release();
			imshow("rst", img);
			waitKey(100);
		}
		else
		{
			cout << "failed to found" << endl;
		}
	}
}