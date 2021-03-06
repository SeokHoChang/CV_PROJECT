package depthPack;

import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_highgui.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;
import static com.googlecode.javacv.cpp.opencv_video.*;
import static com.googlecode.javacv.cpp.opencv_features2d.*;

import java.awt.Color;
import java.awt.Point;
import java.io.IOException;
import java.io.Reader;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

import sun.swing.BakedArrayList;

import com.googlecode.javacpp.FloatPointer;
import com.googlecode.javacpp.IntPointer;
import com.googlecode.javacpp.Loader;
import com.googlecode.javacpp.Pointer;
import com.googlecode.javacpp.PointerPointer;
import com.googlecode.javacpp.ShortPointer;
import com.googlecode.javacv.ObjectFinder;
import com.googlecode.javacv.VideoInputFrameGrabber;
import com.googlecode.javacv.cpp.opencv_core;
import com.googlecode.javacv.cpp.opencv_core.CvContour;
import com.googlecode.javacv.cpp.opencv_core.CvFont;
import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.CvMemStorage;
import com.googlecode.javacv.cpp.opencv_core.CvRect;
import com.googlecode.javacv.cpp.opencv_core.CvSeq;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_features2d.CvSURFPoint;
import com.googlecode.javacv.cpp.opencv_highgui.CvCapture;
import com.googlecode.javacv.cpp.opencv_imgproc.CvMoments;
import com.googlecode.javacv.cpp.opencv_legacy.CvBlob;
import com.googlecode.javacv.cpp.opencv_legacy.CvBlobDetector;
import com.googlecode.javacv.cpp.opencv_legacy.CvBlobSeq;
import com.googlecode.javacv.cpp.opencv_legacy.CvBlobTrack;
import com.googlecode.javacv.cpp.opencv_ml.CvSVM;
import com.sun.org.apache.bcel.internal.generic.FDIV;

import intel.pcsdk.*;

public class main {
	private static final int REALTIME_MODE = 0;
	private static final int TEST_MODE = 1;
	private static final int CAPTURE_MODE = 2;
	

	private static final int SVMSIZE = 6;

	public static final int MATSIZE = 320 * 240;

	private static final int DATSIZE = 576;

	private static final int testNum = 200;
	// /Used measure is Milimeter
	private static final int MAX_DIST = 1000;// mm
	private static final int MIN_SEARCH_RANGE = 270;// mm
	private static final int MAX_SEARCH_RANGE = 500;// mm
	private static final int RANGE = 50;// mm

	private static final int CaptureBox_X = 10;
	private static final int CaptureBox_Y = 10;
	public static final int CaptureBox_WIDTH = 90;
	public static final int CaptureBox_HEIGHT = 90;
	private static final int CaptureBox_MAT_SIZE = 90 * 90;

	public static int cnt = 0;
	private static CvMat dataTest;
	private static CvMat[] dataSets;

	private static short[] depthmap_V, depthmap_R, depthmap_G, depthmap_B;

	private static short[] captureArr;
	private static int[] Size;
	private static IplImage RgbMap, DepthMap, testMap, capture, DepthMap_3C,
			DepthMap_R, DepthMap_G, DepthMap_B, img, MSK;
	private static PXCUPipeline pp;
	private static TestSetMaker testSets;
	private static Classifier classifier;
	private static Converter cvt;

	private static CvSVM[] SVMs;
	private static CvFont font;

	private static boolean isDrawing=false;
	
	// captureBox location
	private static int x = 10, y = 10;
	
	private static int drawX = 0, drawY = 0;
	private static int ClosestX = 0, ClosestY = 0;
	private static int SecondX = 0, SecondY = 0;

	private static double MIN_DIST_VAL = 0;

	private static IplImage DepthImg;
	
	
	
	private static final int confusionMat_row=5;
	private static final int confusionMat_col=5;
	
	private static short[][] confusionMat;
	
	

	public static void init() {

		confusionMat= new short[confusionMat_col][confusionMat_row];
		
		SVMs = new CvSVM[SVMSIZE];
		for (int i = 0; i < SVMSIZE; i++) {
			Classifier svm = new Classifier();
			svm.getSVM().load("SVM_TRAINED" + i, "_0218");
			SVMs[i] = svm.getSVM();
		}

		cvt = new Converter();
		dataSets = new CvMat[2];
		dataTest = cvCreateMat(1, DATSIZE, CV_32FC1);

		for (int j = 0; j < 2; j++) {
			dataSets[j] = cvCreateMat(testNum, DATSIZE, CV_32FC1);
		}
		font = new CvFont();
		cvInitFont(font, CV_FONT_HERSHEY_COMPLEX, 0.7, 0.7, 0, 0, 0);

		classifier = new Classifier();
		cvRect(CaptureBox_X, CaptureBox_Y, CaptureBox_WIDTH, CaptureBox_HEIGHT);
		cvRect(0, 0, CaptureBox_WIDTH, CaptureBox_HEIGHT);

		captureArr = new short[CaptureBox_MAT_SIZE];
		depthmap_V = new short[MATSIZE];
		depthmap_R = new short[MATSIZE];
		depthmap_G = new short[MATSIZE];
		depthmap_B = new short[MATSIZE];
		Size = new int[2];
		DepthImg = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		DepthMap = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		MSK = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		DepthMap_R = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		DepthMap_G = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		DepthMap_B = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);

		testMap = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		capture = cvCreateImage(cvSize(CaptureBox_WIDTH, CaptureBox_HEIGHT),
				IPL_DEPTH_8U, 1);
		DepthMap_3C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		img = cvCreateImage(cvSize(CaptureBox_WIDTH, CaptureBox_HEIGHT),
				IPL_DEPTH_8U, 1);

		cvSetZero(DepthMap);
		cvSetZero(testMap);
		cvSetZero(capture);

		pp = new PXCUPipeline();

		if (!pp.Init(PXCUPipeline.GESTURE)) {

			System.out.print("Failed to initialize PXCUPipeline\n");

			System.exit(3);
		}
	}

	public static void testSet_init() {

		testSets = new TestSetMaker();
		testSets.createTestFile("hand5.dat");
		testSets.recordReady(90, 90);
	}

	public static void makeTestSet(TestSetMaker ts, String file) {
		ts.createTestFile(file);
		ts.recordReady(320, 240);

		for (int k = 0; k < 500; k++) {

			if (!pp.AcquireFrame(true))
				break;

			if (!pp.QueryDepthMap(depthmap_V))
				break;

			pp.QueryDepthMapSize(Size);

			ts.recordTestFrameSet(depthmap_V, 320, 240);

			for (int i = 0; i < depthmap_V.length; i++) {
				if ((depthmap_V[i] < MAX_DIST))
					depthmap_V[i] = (short) ((depthmap_V[i] / (double) MAX_DIST) * 255);

			}

			DepthMap = cvt.CvtArr2Img(DepthMap, depthmap_V, 320, 240);
			
			
			cvNot(DepthMap, DepthMap);
			
			
			cvShowImage("depth", DepthMap);

			pp.ReleaseFrame();
			if (cvWaitKey(1) == 'c')
				break;
			
		}
		ts.recordFinish();

	}

	public static void loadTestSet(String fname, int width, int height,
			int classnum) {
		TestSetMaker ts1 = new TestSetMaker();
		TestSetMaker ts2 = new TestSetMaker();
		ts1.loadReady(fname, width);

		IplImage result = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		short[] depthData = new short[width * height];

		FeatureDescriptor fd0 = new FeatureDescriptor();

		CvFont font = new CvFont();
		CvMat data0;
		cvInitFont(font, CV_FONT_HERSHEY_COMPLEX, 0.5, 0.5, 0, 0, 0);
		int n = 0;

		for (;;) {
			if (n < testNum / 2) {
				depthData = ts1.loadTestFrameSet();

				depthData = Smoothing(img, depthData, CaptureBox_WIDTH,
						CaptureBox_HEIGHT);

				data0 = fd0.MK(depthData, 0, 0, 90, 90);

				// dataSets[0]= correctNAN(data0, dataSets[0]);

				for (int j = 0; j < DATSIZE; j++) {

					dataSets[0].put(n, j, (float) data0.get(0, j));
					if (!(dataSets[0].get(n, j) >= 0)) {
						dataSets[0].put(n, j, 0);
					}

				}
			} else if (n >= testNum / 2 && n < testNum) {
				if (n == testNum / 2) {
					ts2.loadReady("negative" + classnum % 2 + ".dat", width);
				}

				depthData = ts2.loadTestFrameSet();

				depthData = Smoothing(img, depthData, CaptureBox_WIDTH,
						CaptureBox_HEIGHT);

				data0 = fd0.MK(depthData, 0, 0, 90, 90);

				// dataSets[0]= correctNAN(data0, dataSets[0]);

				for (int j = 0; j < DATSIZE; j++) {

					dataSets[0].put(n, j, (float) data0.get(0, j));
					if (!(dataSets[0].get(n, j) >= 0)) {

						dataSets[0].put(n, j, 0);
					}

				}

			} else if (n == testNum) {
				classifier.trainSVM(dataSets[0], testNum, classnum);

			}

			if (n == testNum + 1) {

				int prob = 0;

				for (int i = 0; i < testNum; i++) {
					for (int j = 0; j < DATSIZE; j++)
						dataTest.put(0, j, (float) dataSets[0].get(i, j));

					if (i < testNum / 2)
						if (classifier.getSVM().predict(dataTest, true) == classnum)
							prob++;
					System.out.println(i + " : "
							+ classifier.getSVM().predict(dataTest, false));
				}

				System.out.println("TT Rate=" + (float) prob / testNum);
				System.out.println("TF Rate=" + (1 - (float) prob / testNum));

				classifier.getSVM().save("SVM_TRAINED" + classnum, "_0218");

			}
			if (n > testNum + 1) {
				cvShowImage("datasets", dataSets[0]);
				break;
			}

			result = cvt.CvtArr2Img(img, depthData, width, height);
			cvNot(result, result);
			cvPutText(result, "Cnt:" + n++, cvPoint(10, 20), font,
					CV_RGB(0, 0, 255));
			cvShowImage("depth", result);
			cvWaitKey(1);

			// cvReleaseMat(data0);

		}
		ts1.loadFinish();
		ts2.loadFinish();
	}

	public static int matchHand(CvMat data) {
		MIN_DIST_VAL = 1000;
		int index = -1;
		float minDist = 100;
		float dist = 0;

		for (int i = 0; i < SVMSIZE; i++) {
			// System.out.println(i+" svm:"+SVMs[i].predict(data, true));
			if ((dist = SVMs[i].predict(data, true)) < minDist) {
				cnt++;
				minDist = dist;
				index = i;
			}
		}

		if (minDist >= 0) {
			index = -1;
			// System.out.println(cnt+" duplicate hand");
		} else
			MIN_DIST_VAL = minDist;

		return index;
	}

	public static short[] Smoothing(IplImage img, short[] depthData, int width,
			int height) {
		img = cvt.CvtArr2Img(img, depthData, width, height);

		cvSmooth(img, img, CV_GAUSSIAN, 3);

		return cvt.CvtImg2Arr(img);
	}

	public static void drawCaputeBox(int x, int y) {

		cvDrawCircle(DepthMap_3C, cvPoint(drawX, drawY), 3,
				cvScalar(255, 255, 255, 0), 2, 1, 0);

		x = x < 0 ? 0 : x;
		x = x > 319 - CaptureBox_WIDTH ? 319 - CaptureBox_WIDTH : x;
		y = y < 0 ? 0 : y;
		y = y > 239 - CaptureBox_HEIGHT ? 239 - CaptureBox_HEIGHT : y;

		cvDrawRect(DepthMap_3C, cvPoint(x, y), cvPoint(90 + x, 90 + y),
				cvScalar(0, 255, 0, 0), 2, 0, 0);
		cvDrawCircle(DepthMap_3C, cvPoint(45 + x, 45 + y), 2,
				cvScalar(0, 255, 0, 0), 2, 0, 0);

		cvDrawCircle(DepthMap_3C, cvPoint(45 + x, 45 + y), 36,
				cvScalar(0, 255, 0, 0), 1, 0, 0);
		cvDrawCircle(DepthMap_3C, cvPoint(45 + x, 45 + y), 27,
				cvScalar(0, 255, 0, 0), 1, 0, 0);
		cvDrawCircle(DepthMap_3C, cvPoint(45 + x, 45 + y), 18,
				cvScalar(0, 255, 0, 0), 1, 0, 0);
		cvDrawCircle(DepthMap_3C, cvPoint(45 + x, 45 + y), 9,
				cvScalar(0, 255, 0, 0), 1, 0, 0);
		for (int i = 0; i < 8; i++) {

			cvLine(DepthMap_3C,
					cvPoint(45 + x, 45 + y),
					cvPoint((int) (36 * Math.cos((i * 45 / 180.0) * Math.PI))
							+ 45 + x,
							(int) (36 * Math.sin((i * 45 / 180.0) * Math.PI))
									+ 45 + y), cvScalar(0, 255, 0, 0), 1, 0, 0);
		}

	}

	public static CvMat correctNAN(CvMat data, CvMat dataTest) {
		for (int j = 0; j < DATSIZE; j++) {

			dataTest.put(0, j, (float) data.get(0, j));

			if (!(dataTest.get(0, j) >= 0)) {

				dataTest.put(0, j, 0);
			}

		}
		return dataTest;
	}

	public static short[] findClosestArea(short[] map, short[] depthMap,
			int mode) {
		int ClosestValue = 1000;
		int ClosestIdx = -1;

		for (int i = 0; i < depthMap.length; i++) {

			if (mode == 0) {
				if (depthMap[i] < ClosestValue) {
					ClosestValue = depthMap[i];
					ClosestIdx = i;
				}
			} else {
				if (!(i % 320 >= ClosestX - 45
						&& i % 320 < ClosestX - 45 + CaptureBox_WIDTH
						&& i / 320 >= ClosestY - 45 && i / 320 < ClosestY - 45
						+ CaptureBox_HEIGHT)) {// System.out.println(ClosestValue+","+depthMap[i]);
					if (depthMap[i] < ClosestValue) {
						ClosestValue = depthMap[i];
						ClosestIdx = i;

					}
				}

			}

		}

		if (mode == 0) {
			if (Math.abs(depthMap[ClosestX + ClosestY * 320]
					- depthMap[ClosestIdx % 320 + ClosestIdx / 320 * 320]) > (int) ((30 / (double) MAX_DIST) * 255)) {
				if ((Math.sqrt(Math.pow(ClosestX - ClosestIdx % 320, 2)
						+ Math.pow(ClosestY - ClosestIdx / 320, 2))) > 10) {
					ClosestX = ClosestIdx % 320;
					ClosestY = ClosestIdx / 320;
				}
			}
		} else {
			SecondX = ClosestIdx % 320;
			SecondY = ClosestIdx / 320;

		}

		int min = ClosestValue;
		int max = min + (int) ((130 / (double) MAX_DIST) * 255);

		for (int i = 0; i < depthMap.length; i++) {
			int X = i % 320;
			int Y = i / 320;
			double r = 0;
			if (mode == 0)
				r = Math.sqrt(Math.pow(X - ClosestX, 2)
						+ Math.pow(Y - ClosestY, 2));
			else
				r = Math.sqrt(Math.pow(X - SecondX, 2)
						+ Math.pow(Y - SecondY, 2));

			if (((depthMap[i] < max && depthMap[i] > min) && (r < 95))) {
				short v = depthMap[i];
				map[i] = v;
			} else
				map[i] = 255;
		}

		return map;

	}

	public static void MEANShift(short[] depthMap, int originX, int originY,
			int mode) {
		double d = 1000;
		int Xsum = 0;
		int Ysum = 0;
		int n = 1;

		int meanX = 0;
		int meanY = 0;

		while (!(d < 5)) {
			for (int i = 0; i < depthMap.length; i++) {

				if (depthMap[i] != 255) {
					n++;
					Xsum += i % 320;
					Ysum += i / 320;

				}
			}
			meanX = Xsum / n;
			meanY = Ysum / n;

			d = Math.sqrt(Math.pow(meanX - originX, 2)
					+ Math.pow(meanY - originY, 2));
			originX = meanX;
			originY = meanY;
			Xsum = 0;
			Ysum = 0;
			n = 1;
			// System.out.println(d);

		}
		if (mode == 0) {
			ClosestX = meanX;
			ClosestY = meanY;
		} else {
			SecondX = meanX;
			SecondY = meanY;
		}

	}

	public static CvSeq findBiggestContour(CvSeq contours) {
		CvSeq MaxContourPtr = null;
		CvSeq MaxContourPtr1 = null;
		CvRect contourBox = null;
		CvRect contourBox1 = null;
		int boxArea = 0;
		int maxArea = -1;

		for (CvSeq ptr = contours; ptr != null; ptr = ptr.h_next()) {

			contourBox = cvBoundingRect(ptr, 1);

			boxArea = contourBox.width() * contourBox.height();

			if (boxArea < 45 * 45 // || boxArea>100*100
			)
				continue;

			if (boxArea > maxArea) {

				// cvDrawRect(DepthMap_3C, cvPoint(contourBox.x(),
				// contourBox.y()),cvPoint(contourBox.x()+contourBox.width(),
				// contourBox.y()+contourBox.height()), cvScalar(0,255, 0, 0),
				// 2,0, 0);

				maxArea = boxArea;
				MaxContourPtr = ptr;
			}

		}

		// if(contourBox!=null)
		// cvDrawRect(DepthMap_3C, cvPoint(contourBox.x(),
		// contourBox.y()),cvPoint(contourBox.x()+contourBox.width(),
		// contourBox.y()+contourBox.height()), cvScalar(0,255, 0, 0), 2,0, 0);
		//

		return MaxContourPtr;

	}

	public static void findHand(IplImage DepthImg) {
		cvThreshold(DepthImg, MSK, 0, 255, CV_THRESH_BINARY);

		CvMemStorage mem = cvCreateMemStorage(0);
		CvSeq contours = new CvSeq();
		CvSeq Maxcontour;

		cvShowImage("MSK", MSK);
		int contourNum = cvFindContours(MSK, mem, contours,
				Loader.sizeof(CvContour.class), CV_RETR_LIST,
				CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

		if (contourNum > 0) {
			Maxcontour = findBiggestContour(contours);

			if (Maxcontour != null) {

				CvMemStorage storage = cvCreateMemStorage(0);

				CvSeq poly = cvApproxPoly(Maxcontour,
						Loader.sizeof(CvContour.class), storage,
						CV_POLY_APPROX_DP, 3, 1);

				cvDrawContours(DepthMap_3C, poly, cvScalar(255, 0, 0, 0),
						cvScalar(0, 0, 0, 0), 1, 3, 8);

				cvReleaseMemStorage(storage);
			}

		}
		cvReleaseMemStorage(mem);

	}

	public static short[] maskDepthMap(short[] depthmap, int centerX,
			int centerY) {
		int minA = depthmap[ClosestX + ClosestY * 320];
		int maxA = minA + (int) ((150 / (double) MAX_DIST) * 255);

		int minB = depthmap[SecondX + SecondY * 320];
		int maxB = minB + (int) ((130 / (double) MAX_DIST) * 255);

		for (int i = 0; i < depthmap.length; i++) {
			boolean ishand1 = i % 320 >= centerX
					&& i % 320 < centerX + CaptureBox_WIDTH
					&& i / 320 >= centerY
					&& i / 320 < centerY + CaptureBox_HEIGHT
			// &&(depthmap[i]<maxA && depthmap[i]>minA)
			;
			// boolean ishand2
			// = i%320>=SecondX-CaptureBox_WIDTH/2 &&
			// i%320<SecondX-CaptureBox_WIDTH/2+CaptureBox_WIDTH &&
			// i/320>=SecondY-CaptureBox_HEIGHT/2 &&
			// i/320<SecondY-CaptureBox_HEIGHT/2 + CaptureBox_HEIGHT
			// //&&(depthmap[i]<maxB&&depthmap[i]>minB)
			// ;

			if (!(ishand1)) {
				depthmap[i] = 255;

			}

		}
		return depthmap;
	}

	public static void realTimeShow(int mode) {
		CvCapture cap1 = cvCreateCameraCapture(0);

		FeatureDescriptor fd = new FeatureDescriptor();
		ImageController imgCont = new ImageController();
		GUI gui = new GUI();
		boolean Chkmatch = false;
		boolean hold = true;

		int cnt = 0;
		boolean Fin = false;

		TestSetMaker test = new TestSetMaker();
		

		if (mode == TEST_MODE)
			test.loadReady("TestFrame1.dat", 320);
		
		
		while (!Fin) {
			RgbMap = cvQueryFrame(cap1);

			if (mode == REALTIME_MODE) {
				if (!pp.AcquireFrame(true))
					break;

				pp.QueryDepthMapSize(Size);

				if (!pp.QueryDepthMap(depthmap_V))
					break;
			} else if (mode == TEST_MODE) {
				depthmap_V = test.loadTestFrameSet();
				if(depthmap_V==null)
					break;
			} else {
				System.out.println("Wrong MODE NUMBER");

				break;
			}

			for (int i = 0; i < depthmap_V.length; i++) {

				depthmap_B[i] = 0;
				depthmap_G[i] = 0;
				depthmap_R[i] = 0;

				if ((depthmap_V[i] < MAX_DIST)) {
					depthmap_V[i] = (short) Math
							.round(((depthmap_V[i] / (double) MAX_DIST) * 255));

					if (depthmap_V[i] < 130)
						depthmap_R[i] = (short) ((depthmap_V[i] / (double) 130) * 255);

					else if (130 < depthmap_V[i] && depthmap_V[i] < 200)
						depthmap_G[i] = (short) (depthmap_V[i] / (double) 200 * 255);
					else if (200 < depthmap_V[i] && depthmap_V[i] < 255)
						depthmap_B[i] = (short) (depthmap_V[i] / (double) 255 * 255);

				} else
					depthmap_V[i] = 255;
			}
			DepthMap_R = cvt.CvtArr2Img(DepthMap_R, depthmap_R, 320, 240);
			DepthMap_G = cvt.CvtArr2Img(DepthMap_G, depthmap_G, 320, 240);
			DepthMap_B = cvt.CvtArr2Img(DepthMap_B, depthmap_B, 320, 240);
			DepthMap = cvt.CvtArr2Img(DepthMap, depthmap_V, 320, 240);

			cvNot(DepthMap, DepthMap);
			cvMerge(DepthMap_B, DepthMap_G, DepthMap_R, null, DepthMap_3C);

			short[] map = new short[MATSIZE];
			map = findClosestArea(map, depthmap_V, 0);

			drawX=ClosestX;
			drawY=ClosestY;
			
			MEANShift(map, ClosestX, ClosestY, 0);
			drawCaputeBox(ClosestX - 45, ClosestY - 55);

			// short[] map1= new short[MATSIZE];
			// if((map1= findClosestArea(map1,depthmap_V,1))!=null)
			// {
			// MEANShift(map1, SecondX, SecondY,1);
			// drawCaputeBox(SecondX-45, SecondY-45);
			// }

			map = maskDepthMap(map, ClosestX - 45, ClosestY - 55);
			DepthImg = cvt.CvtArr2Img(DepthImg, map, 320, 240);
			cvNot(DepthImg, DepthImg);
			//findHand(DepthImg);
			
			if(mode==TEST_MODE)
				Chkmatch=true;
			if(mode == CAPTURE_MODE )
				Chkmatch=false;
			
			
			int handNum=-1;
			if (Chkmatch == true) {
				captureArr = captureBoxImage(DepthMap, capture, ClosestX - 45,
						ClosestY - 55, 0);
				captureArr = Smoothing(img, captureArr, 90, 90);

				dataTest = correctNAN(fd.MK(captureArr, 0, 0, 90, 90), dataTest);

//				int i = fd.getDominanAngleIdx();
//				// System.out.println((360-45*i)+"~"+(360-45*(i+1)));
//
//				cvLine(DepthMap_3C,
//						cvPoint(ClosestX, ClosestY - 10),
//						cvPoint((int) (36 * Math
//								.cos((i * 45 / 180.0) * Math.PI)) + ClosestX,
//								(int) (36 * Math
//										.sin((i * 45 / 180.0) * Math.PI))
//										+ ClosestY - 10),
//						cvScalar(255, 0, 0, 0), 3, 0, 0);

				// capture=imgCont.setImageOrientation(capture, i);
				// System.out.println(i);

				cvShowImage("CaptureBox", capture);
			
				 handNum= matchHand(dataTest);
				
				

				if (mode != TEST_MODE) {					
					gui.setDrawPoint(320-drawX, drawY);
					gui.setHandPosition(320-ClosestX, ClosestY - 10);
					gui.countMoving();
					
					gui.getHandNum(handNum);
					cvPutText(DepthMap_3C,gui.getState()+ "", cvPoint(15, 15), font,
							CV_RGB(0,180, 0));
					gui.paint();

				}

				if (handNum != -1)
					cvPutText(DepthMap_3C, handNum + "", cvPoint(45, 45), font,
							CV_RGB(255, 0, 0));

				
			
				
			}
			switch (cvWaitKey(1)) {

			case 'g':
			case 'G':	
				Chkmatch = (Chkmatch == true) ? false : true;
				break;
			case 'c':
			case 'C':
				System.out.println(cnt++);
				captureBoxImage(DepthMap, capture, ClosestX - 45,
						ClosestY - 55, 1);
				cvShowImage("CaptureBox", capture);
				break;
			case 's':
			case 'S':
				testSets.recordFinish();
				System.out.println("record finish");
				System.exit(1);
				break;

			case 27:
				Fin = true;
				break;
			}

			cvShowImage("depth3C", DepthMap_3C);
			cvShowImage("depth1C", DepthMap);
			cvShowImage("RGB", RgbMap);
			
			
			if(mode==TEST_MODE)
			{
				chkTF(handNum);
			}
			
			
			pp.ReleaseFrame();

		}

		if (mode == TEST_MODE){
			test.loadFinish();
			printConfusionMat();
		}
		
		cvReleaseImage(DepthMap);
		cvReleaseImage(DepthMap_3C);
		cvReleaseImage(RgbMap);
		cvReleaseImage(MSK);

	}

	public static short[] captureBoxImage(IplImage DepthMap, IplImage capture,
			int x, int y, int mode) {

		x = x < 0 ? 0 : x;
		x = x > 319 - CaptureBox_WIDTH ? 319 - CaptureBox_WIDTH : x;
		y = y < 0 ? 0 : y;
		y = y > 239 - CaptureBox_HEIGHT ? 239 - CaptureBox_HEIGHT : y;

		cvRect(x, y, CaptureBox_WIDTH, CaptureBox_HEIGHT);

		cvSetImageROI(DepthMap,
				cvRect(x, y, CaptureBox_WIDTH, CaptureBox_HEIGHT));
		cvCopy(DepthMap, capture);
		cvResetImageROI(DepthMap);

		for (int i = 0, j = 0; i < MATSIZE; i++) {
			if ((i % 320) >= x && (i % 320) < CaptureBox_WIDTH + x
					&& (i / 320) >= y && (i / 320) < CaptureBox_HEIGHT + y) {
				captureArr[j++] = depthmap_V[i];
			}
		}

		// / save Frame continuously
		if (mode == 1)
			testSets.recordTestFrameSet(captureArr, CaptureBox_WIDTH,
					CaptureBox_HEIGHT);

		return captureArr;
	}

	public static void chkTF(int handNum)
	{
		
			int ans= cvWaitKey(10000);
			
			switch(ans)
			{
			case '0':
				ans=0;
				break;
			case '1':
				ans=1;
				break;
			case '2':
				ans=2;
				break;
			case '3':
				ans=3;
				break;
			case 'f':
				ans=-1;
				break;
			default:
				return;
				
			}
			System.out.println("Answer: " + ans);
			System.out.println("Machin said :"+ handNum);
		
			int i=handNum,j=ans;
			if(ans==-1)
				j=4;
			if(handNum==-1)
				i=4;
			
		confusionMat[i][j]+=1;
		
	}
	
	
	public static void printConfusionMat()
	{
		for (int i = 0; i < confusionMat_row; i++) {
			for (int j = 0; j < confusionMat_col; j++) {
				System.out.print(confusionMat[i][j]+" ");
			}
			System.out.println();
		}
	}
	public static void main(String[] args) {

		init();
		// testSet_init();

		//PPT2IMG p2i = new PPT2IMG("컴퓨터비전.pptx", "컴퓨터비전");
		//try {
		//	p2i.converter();
	//	} catch (IOException e) {
	//		// TODO Auto-generated catch block
	//		e.printStackTrace();
	//	}
	//	System.out.println("yes JAM");

	    //makeTestSet(new TestSetMaker(),"TestFrame1.dat");
		//
//		 loadTestSet("hand0.dat",90,90,0);
//		 loadTestSet("hand1.dat",90,90,1);
//		 loadTestSet("hand2.dat",90,90,2);
//		 loadTestSet("hand3.dat",90,90,3);
//		 loadTestSet("hand4.dat",90,90,4);
		// loadTestSet("hand5.dat",90,90,5);
		//realTimeShow(TEST_MODE);
		  realTimeShow(REALTIME_MODE);
		//  realTimeShow(CAPTURE_MODE);
		//
		//
		//
		pp.Close();
		//
		System.exit(0);

	}

}
