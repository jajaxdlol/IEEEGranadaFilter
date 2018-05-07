#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void put_moustache(Mat mst, Mat* image, Rect faces) {
	
	int face_width = faces.width;
	int face_height = faces.height;
	int x = faces.x;
	int y = faces.y;

	

	vector<Mat> image_ch;
	split(*image, image_ch);

	Mat mstr;
	int mst_width = int(face_width*0.7166666) + 1;
	int mst_height = int(face_height*0.142857) + 1;

	resize(mst, mstr, Size(mst_width, mst_height));

	vector<Mat> mst_ch;
	split(mstr, mst_ch);

	for (int i = int(0.62857142857*face_height); i < int(0.62857142857*face_height) + mst_height; i++) {
		for (int j = int(0.14166666666*face_width); j < int(0.14166666666*face_width) + mst_width; j++) {
			if (int(mst_ch[0].at<uchar>(i - int(0.62857142857*face_height), j - int(0.14166666666*face_width))) < 235)
				image_ch[0].at<uchar>(y + i, x + j) = mst_ch[0].at<uchar>(i - int(0.62857142857*face_height), j - int(0.14166666666*face_width));

			if (int(mst_ch[1].at<uchar>(i - int(0.62857142857*face_height), j - int(0.14166666666*face_width))) < 235)
				image_ch[1].at<uchar>(y + i, x + j) = mst_ch[1].at<uchar>(i - int(0.62857142857*face_height), j - int(0.14166666666*face_width));

			if (int(mst_ch[2].at<uchar>(i - int(0.62857142857*face_height), j - int(0.14166666666*face_width))) < 235)
				image_ch[2].at<uchar>(y + i, x + j) = mst_ch[2].at<uchar>(i - int(0.62857142857*face_height), j - int(0.14166666666*face_width));
		}
	}

	merge(image_ch, *image);
	rectangle(*image, faces, Scalar(255, 0, 0));
}

void put_hat(Mat hat, Mat* image, Rect faces) {

	int face_width = faces.width;
	int face_height = faces.height;
	int x = faces.x;
	int y = faces.y;



	vector<Mat> image_ch;
	split(*image, image_ch);

	Mat hatr;
	int hat_width = face_width +1;
	int hat_height = int(face_height*0.35) + 1;

	resize(hat, hatr, Size(hat_width, hat_height));

	vector<Mat> hat_ch;
	split(hatr, hat_ch);

	for (int i = 0; i < hat_height; i++) {
		for (int j = 0; j < hat_width; j++) {
			if (int(hat_ch[0].at<uchar>(i , j)) < 235)
				image_ch[0].at<uchar>(y + i - int(0.25*face_height), x + j) = hat_ch[0].at<uchar>(i, j);

			if (int(hat_ch[1].at<uchar>(i, j)) < 235)
				image_ch[1].at<uchar>(y + i - int(0.25*face_height), x + j) = hat_ch[1].at<uchar>(i, j);

			if (int(hat_ch[2].at<uchar>(i, j)) < 235)
				image_ch[2].at<uchar>(y + i - int(0.25*face_height), x + j) = hat_ch[2].at<uchar>(i, j);
		}
	}

	merge(image_ch, *image);
	rectangle(*image, faces, Scalar(255, 0, 0));
}

int main() {
	char cascPath[] = "haarcascade_frontalcatface.xml";
	CascadeClassifier face_cascade;
	vector<Rect> faces;
	Rect face;
	VideoCapture capture;
	Mat frame;
	Mat frameg;
	Mat mst;
	Mat hat;
	Mat dog;

	mst = imread("moustache.png");
	hat = imread("cowboy_hat.png");

	char choice;

	//-- 1. Load the cascades
	if (!face_cascade.load(cascPath)) { 
		printf("--(!)Error loading\n"); 
		cvWaitKey(0);
		return -1; };

	capture.open(0);
	if (!capture.isOpened()) {
		cout << "No webcam input" << endl;
		return 1;
	}

	cout << "Elige, ponerte un buen bigottenn (A), o ponerte un sombrero de vaquero (B): ";
	cin >> choice;
	while (1) {
		if (!capture.read(frame))
			continue;
		cvtColor(frame, frameg, CV_BGR2GRAY);
		face_cascade.detectMultiScale(frameg, faces, 1.1, 3, 0| CV_HAAR_SCALE_IMAGE, Size(40, 40));
		cout << faces.size() << endl;
		if (faces.size() > 0){
			face = faces.at(0);
			switch (choice)
			{
			case 'A': 
				put_moustache(mst, &frame, face);
				break;

			case 'B':
				put_hat(hat, &frame, face);
				break;
			}
		}
		imshow("Video", frame);
		imshow("VideoBN", frameg);
		cvWaitKey(2);
	}
	cvWaitKey(0);
	

}
