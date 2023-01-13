#include <iostream>
#include<opencv2\objdetect\objdetect.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include "opencv2\core.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include "opencv2/face.hpp"
#include <opencv2/face.hpp>
#include <fstream>
#include <sstream>


using namespace std;
using namespace cv;
using namespace cv::face;

vector<string> names;
vector<int> amount_fotos;


Mat add_face()
{
	Mat frame, gray, face_frame, res;
	CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
	int responce;

	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		cout << "Camera fail!!!\n";
		frame.data = NULL;
		return frame;
	}

	while (capture.read(frame))
	{
		vector<Rect> faces;

		cvtColor(frame, gray, COLOR_BGR2GRAY);

		faceDetector.detectMultiScale(gray, faces);

		for (int i = 0; i < faces.size(); i++)
		{
			face_frame = gray(faces[i]);
			cv::rectangle(frame, faces[i], Scalar(0, 255, 0), 3);
		}
		
		//resize(frame, res, Size(240, 240), 0, 0, INTER_LINEAR);

		imshow("Face", gray);
		cout << "Save frame? 1. Yes 2. No\n";
		cin >> responce;

		if (responce == 1)
		{
			return face_frame;			
		}
		else
		{
			continue;
		}
		if (waitKey(1) == 27)
		{
			break;
		}
		
	}

	return face_frame;
}

static Mat norm_0_255(InputArray _src) 
{
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}

int train_eigenface(const char* path)
{
	string name;
	int response, amount_foto, flag, index, foto_index;
	Mat face_frame;

	cout << "Enter your name:\n";
	cin >> name;

	flag = 0;
	for (int i = 0; i < names.size(); i++)
	{
		if (name == names[i])
		{
			cout << "This person is already in the database!\n";
			cout << "Continue? 1. Yes 2. No\n";
			cin >> response;
			if (response==1)
			{
				flag = 1;
				index = i;
				break;				
			}
			else
			{
				cout << "Enter a unique name!\n";
				cin >> name;
				i = 0;
				continue;
			}
		}
	}
	
	cout << "how many photos to add?\n";
	cin >> amount_foto;
	stringstream path_file;
	for (int i = 0; i < amount_foto; i++)
	{
		face_frame = add_face();
		imshow("Face", face_frame);
		string filename = path;
		//filename = filename + "\\" + name + "\\image (";

		if (flag == 0)
		{
			names.push_back(name);
			foto_index = i;
		}
		else
		{
			foto_index = amount_fotos[index] + i;
		}
		//filename = filename + foto_index;
		path_file << path << "\\" << name << "\\image (" << foto_index << ").jpg";
		filename = path_file.str();
		path_file.str("");
		path_file.clear();
		//path_file >> filename;		
		cout << filename << '\n';
		imwrite(filename, face_frame);
		
	}

	


	return 0;
}

int face_recognize()
{
	CascadeClassifier faceDetector("haarcascade_frontalface_alt2.xml");
	Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();
	//Ptr<FaceRecognizer>  model = FisherFaceRecognizer::create();

	model->read("C:\\Users\\Бобур Ибрагимов\\source\\repos\\EigenFace_recognize\\eigenface.yml");

	//Ptr<Facemark> facemark = FacemarkLBF::create();
	//facemark->loadModel("lbfmodel.yaml");

	VideoCapture cam(0);
	Mat frame, gray, resize_frame, threshold_frame, image;
	int label;

	image = imread("E:\\Eigen train\\I_1\\image (1).jpg", IMREAD_GRAYSCALE);

	resize(image, image, Size(300, 400), 1.0, 1.0, INTER_CUBIC);

	label = model->predict(image);
	cout << label << '\n';

	image = imread("E:\\Eigen train\\Azim-aka_1\\image (1).jpg", IMREAD_GRAYSCALE);
	resize(image, image, Size(300, 400), 1.0, 1.0, INTER_CUBIC);
	label = model->predict(image);
	cout << label << '\n';

	image = imread("E:\\Eigen train\\Aziz_1\\image (1).jpg", IMREAD_GRAYSCALE);
	resize(image, image, Size(300, 400), 1.0, 1.0, INTER_CUBIC);
	label = model->predict(image);
	cout << label << '\n';

	/*Mat eigenvalues = model->getEigenValues();

	Mat W = model->getEigenVectors();

	Mat mean = model->getMean();

	//imshow("mean", norm_0_255(mean.reshape(1, image.rows)));

	int height = image.rows;

	for (int i = 0; i < min(80, W.cols); i++)
	{
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		//cout << msg << endl;
		// get eigenvector #i
		Mat ev = W.col(i).clone();
		// Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale = norm_0_255(ev.reshape(1, height));
		// Show the image & apply a Jet colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
		// Display or save:

		//imshow(format("eigenface_%d", i), cgrayscale);
		imwrite(format("%s/eigenface_%d.png", "E:\\Eigen train\\Test", i), norm_0_255(cgrayscale));
	}

	for (int num_components = min(W.cols, 10); num_components < min(W.cols, 300); num_components += 15)
	{
		// slice the eigenvectors from the model
		Mat evs = Mat(W, Range::all(), Range(0, num_components));
		Mat projection = LDA::subspaceProject(evs, mean, image.reshape(1, 1));
		Mat reconstruction = LDA::subspaceReconstruct(evs, mean, projection);
		// Normalize the result:
		reconstruction = norm_0_255(reconstruction.reshape(1, image.rows));
		// Display or save:

		imwrite(format("%s/eigenface_reconstruction_%d.png", "E:\\Eigen train\\Test\\new", num_components), reconstruction);
	}*/

	if (!cam.isOpened())
	{
		cout << "exit" << endl;
		return 1;
	}

	while (cam.read(frame))
	{
		vector<Rect> faces;


		cvtColor(frame, gray, COLOR_BGR2GRAY);


		faceDetector.detectMultiScale(gray, faces);

		//GaussianBlur(gray, blur, Size(5, 5), 0, 0);

		//vector< vector<Point2f> > landmarks;
		// Run landmark detector
		//bool success = facemark->fit(frame, faces, landmarks);

		//cout << faces.size() << '\n';

		for (int i = 0; i < faces.size(); i++)
		{
			resize_frame = gray(faces[i]);
			//resize_frame = blur(faces[i]);
			cv::rectangle(frame, faces[i], Scalar(0, 255, 0), 3);

			//cv::resize(resize_frame, resize_frame, Size(150, 200), 1.0, 1.0, INTER_CUBIC);
			cv::resize(resize_frame, resize_frame, Size(300, 400), 1.0, 1.0, INTER_CUBIC);

			//threshold(resize_frame, threshold_frame, 0, 255, THRESH_OTSU | THRESH_BINARY_INV);

			int predicted_label = -1;
			double predicted_confidence = 0.0;
			//cout << "width is: " << faces[i].width << '\n';
			//cout << "height is: " << faces[i].height << '\n';

			//imshow("Face Recognize", resize_frame);

			model->predict(resize_frame, predicted_label, predicted_confidence);
			//predicted_label = model->predict(resize_frame);

			//cout << "label is: " << predicted_label << '\n';
			//cout << "confidence is: " << predicted_confidence << '\n';


			switch (predicted_label)
			{
			case 0:
				putText(frame, "Bobur", Point(faces[i].x, faces[i].y), 1, 2, Scalar(0, 0, 255), 2);
				break;
			case 1:
				putText(frame, "Azim-aka", Point(faces[i].x, faces[i].y), 1, 2, Scalar(0, 0, 255), 2);
				break;
			case 2:
				putText(frame, "Aziz", Point(faces[i].x, faces[i].y), 1, 2, Scalar(0, 0, 255), 2);
				break;
			default:
				putText(frame, "Unknown person", Point(faces[i].x, faces[i].y), 1, 2, Scalar(0, 0, 255), 2);
				break;
			}
		}

		imshow("Face Recognize", frame);

		// Exit loop if ESC is pressed
		if (waitKey(1) == 27)
			break;

	}

	return 0;
}

int main(int argc, char** argv)
{
	ifstream base;
	int i, size_name_vec, size_foto, choice;
	string buff, addr;

	base.open("Base.txt");
	base >> size_name_vec;
	//cout << size_name_vec << '\n';
	for ( i = 0; i < size_name_vec; i++)
	{
		base >> buff;
		names.push_back(buff);
		//cout << names[i] << '\n';
	}
	for (i = 0; i < size_name_vec; i++)
	{
		base >> size_foto;
		amount_fotos.push_back(size_foto);
		//cout << amount_fotos[i] << '\n';
	}
	base >> addr;
	//cout << addr << '\n';

	base.close();


	while (true)
	{
		cout << "1. Recognise Face\n";
		cout << "2. Add Face\n";
		cout << "3. Exit\n";
		cout << "Choose One: ";
		cin >> choice;
		switch (choice)
		{
		case 1:
			face_recognize();
			break;
		case 2:
			//add_face();
			train_eigenface(addr.c_str());
			break;
		case 3:
			return 0;
			break;
		default:
			return 0;
		}
	}
	


	return 0;
}
