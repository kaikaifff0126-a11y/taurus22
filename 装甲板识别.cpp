#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iomanip>

using namespace std;
using namespace cv;

constexpr double LIGHTBAR_LENGTH = 56e-3;
constexpr double SMALL_ARMOR_WIDTH = 135e-3;

const vector<Point3f> SMALL_ARMOR_POINTS{
    {0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
    {0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
    {0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
    {0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}
};

double getLeanAngle(const RotatedRect& box)
{
    Point2f points[4];
    box.points(points);

    Point2f l1 = points[1] - points[0];
    Point2f l2 = points[1] - points[2];

    Point2f direction;

    if (norm(l1) > norm(l2))
        direction = l1;
    else
        direction = l2;

    if (direction.y < 0)
    {
        direction.x = -direction.x;
        direction.y = -direction.y;
    }

    return atan2(direction.x, direction.y) * 180.0 / CV_PI;
}

void getLightEndpoints(
    const RotatedRect& box,
    Point2f& top,
    Point2f& bottom)
{
    Point2f points[4];
    box.points(points);

    Point2f edge1 = points[1] - points[0];
    Point2f edge2 = points[2] - points[1];

    Point2f p1, p2;

    if (norm(edge1) > norm(edge2))
    {
        p1 = (points[0] + points[3]) * 0.5f;
        p2 = (points[1] + points[2]) * 0.5f;
    }
    else
    {
        p1 = (points[0] + points[1]) * 0.5f;
        p2 = (points[2] + points[3]) * 0.5f;
    }

    if (p1.y < p2.y)
    {
        top = p1;
        bottom = p2;
    }
    else
    {
        top = p2;
        bottom = p1;
    }
}

vector<Point2f> getArmorPoints(
    const RotatedRect& a,
    const RotatedRect& b)
{
    RotatedRect left = a;
    RotatedRect right = b;

    if (left.center.x > right.center.x)
        swap(left, right);

    Point2f leftTop, leftBottom;
    Point2f rightTop, rightBottom;

    getLightEndpoints(left, leftTop, leftBottom);
    getLightEndpoints(right, rightTop, rightBottom);

    return {
        leftTop,
        leftBottom,
        rightBottom,
        rightTop
    };
}

int main()
{
    Scalar lower(0, 45, 63);
    Scalar upper(110, 240, 255);

    VideoCapture cap(
        "C:/TestVideo/8.4第二场_clip_194.800_204.avi"
    );

    if (!cap.isOpened())
    {
        cout << "Failed to open video." << endl;
        return -1;
    }

    Mat CameraMatrix;
    Mat diffcoeffs;

    FileStorage fs("C:/cameracaliberate/camera.yml", FileStorage::READ);

    if (!fs.isOpened())
    {
        cout << "Failed to open calibration file." << endl;
        return -1;
    }

    fs["CameraMatrix"] >> CameraMatrix;
    fs["diffcoeffs"] >> diffcoeffs;
    fs.release();

    if (CameraMatrix.empty() || diffcoeffs.empty())
    {
        cout << "Invalid calibration parameters." << endl;
        return -1;
    }

    CameraMatrix.convertTo(CameraMatrix, CV_64F);
    diffcoeffs.convertTo(diffcoeffs, CV_64F);

    Mat kernel = getStructuringElement(
        MORPH_RECT, Size(5, 5)
    );

    Mat frame;

    while (1)
    {
        if (!cap.read(frame))
            break;

        Mat imgClone = frame.clone();
        Mat imgHSV, imgBlur, mask, maskOpen;

        vector<vector<Point>> contours;
        vector<RotatedRect> lights;

        GaussianBlur(frame, imgBlur, Size(3, 3), 0);
        cvtColor(imgBlur, imgHSV, COLOR_BGR2HSV);
        inRange(imgHSV, lower, upper, mask);

        morphologyEx(
            mask, maskOpen,
            MORPH_OPEN, kernel
        );

        findContours(
            maskOpen, contours,
            RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
        );

        for (size_t i = 0; i < contours.size(); i++)
        {
            RotatedRect box = minAreaRect(contours[i]);

            Point2f points[4];
            box.points(points);

            float longside = max(
                box.size.height, box.size.width
            );

            float shortside = min(
                box.size.height, box.size.width
            );

            if (shortside <= 0)
                continue;

            float ratio = longside / shortside;

            if (ratio < 2.0 || ratio > 5.0)
                continue;

            Point2f v1 = points[1] - points[0];
            Point2f v2 = points[2] - points[1];

            Point2f direction;

            if (norm(v1) > norm(v2))
                direction = v1;
            else
                direction = v2;

            double tilt =
                atan2(abs(direction.x), abs(direction.y))
                * 180.0 / CV_PI;

            if (tilt > 30)
                continue;

            for (int j = 0; j < 4; j++)
            {
                line(
                    imgClone,
                    points[j],
                    points[(j + 1) % 4],
                    Scalar(0, 255, 0),
                    2
                );
            }

            lights.push_back(box);
        }

        for (size_t i = 0; i < lights.size(); i++)
        {
            for (size_t j = i + 1; j < lights.size(); j++)
            {
                RotatedRect a = lights[i];
                RotatedRect b = lights[j];

                double lengthA = max(
                    a.size.height, a.size.width
                );

                double lengthB = max(
                    b.size.height, b.size.width
                );

                double ratio =
                    max(lengthA, lengthB) /
                    min(lengthA, lengthB);

                double heightdiff =
                    abs(a.center.y - b.center.y);

                double spacingdiff =
                    abs(a.center.x - b.center.x);

                double average =
                    (lengthA + lengthB) / 2.0;

                double heightRatio =
                    heightdiff / average;

                double spacingRatio =
                    spacingdiff / average;

                if (ratio > 1.5)
                    continue;

                if (heightRatio > 0.5)
                    continue;

                if (spacingRatio < 1.0 ||
                    spacingRatio > 5.0)
                    continue;

                double angleA = getLeanAngle(a);
                double angleB = getLeanAngle(b);

                double angleDiff =
                    abs(angleA - angleB);

                if (angleDiff > 10.0)
                    continue;

                vector<Point2f> imagePoints =
                    getArmorPoints(a, b);

                Mat rvec, tvec;

                bool success = solvePnP(
                    SMALL_ARMOR_POINTS,
                    imagePoints,
                    CameraMatrix,
                    diffcoeffs,
                    rvec,
                    tvec,
                    false,
                    SOLVEPNP_ITERATIVE
                );

                if (!success)
                    continue;

                double x = tvec.at<double>(0);
                double y = tvec.at<double>(1);
                double z = tvec.at<double>(2);

                double distance = sqrt(
                    x * x + y * y + z * z
                );

                Rect armorBox = boundingRect(imagePoints);

                rectangle(
                    imgClone,
                    armorBox,
                    Scalar(0, 255, 255),
                    2
                );

                Point armorCenter(
                    armorBox.x + armorBox.width / 2,
                    armorBox.y + armorBox.height / 2
                );

                circle(
                    imgClone,
                    armorCenter,
                    4,
                    Scalar(0, 0, 255),
                    -1
                );

                string distanceText =
                    cv::format("Distance: %.2f m", distance);

                string xyzText =
                    cv::format(
                        "X: %.2f Y: %.2f Z: %.2f",
                        x, y, z
                    );

                putText(
                    imgClone,
                    distanceText,
                    Point(
                        armorBox.x,
                        max(armorBox.y - 25, 20)
                    ),
                    FONT_HERSHEY_SIMPLEX,
                    0.6,
                    Scalar(0, 255, 255),
                    2
                );

                putText(
                    imgClone,
                    xyzText,
                    Point(
                        armorBox.x,
                        max(armorBox.y - 5, 40)
                    ),
                    FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Scalar(255, 255, 0),
                    2
                );

                cout << fixed << setprecision(3)
                    << "X: " << x
                    << " Y: " << y
                    << " Z: " << z
                    << " Distance: " << distance
                    << " m" << endl;
            }
        }

        imshow("original", frame);
        imshow("mask", maskOpen);
        imshow("rst", imgClone);

        if (waitKey(30) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
