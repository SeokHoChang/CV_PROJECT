package depthPack;


import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_highgui.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;
import static com.googlecode.javacv.cpp.opencv_video.*;
import static com.googlecode.javacv.cpp.opencv_features2d.*;

import java.awt.Color;
import java.awt.Point;
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

	private static final int SVMSIZE = 4;
	
	public static final int MATSIZE = 320*240;
	
	private static final int DATSIZE = 576;
	
	private static final int testNum= 180;
	///Used measure is Milimeter
	private static final int MAX_DIST = 1000;//mm
	private static final int MIN_SEARCH_RANGE=270;//mm
	private static final int MAX_SEARCH_RANGE=500;//mm
	private static final int RANGE=50;//mm
	 
	private static final int CaputureBox_X=10;
	private static final int CaputureBox_Y=10;
	public static final int CaputureBox_WIDTH=90;
	public static final int CaputureBox_HEIGHT=90;
	private static final int CaputureBox_MAT_SIZE=90*90;

	public static int cnt=0;
	private static CvMat dataTest;
	private static CvMat[] dataSets;
	
	private static short[] depthmap_V,depthmap_R,depthmap_G,depthmap_B;

	private static short[] captureArr;
	private static int[] Size;
	private static IplImage RgbMap,DepthMap,testMap,capture,DepthMap_3C,DepthMap_R,DepthMap_G,DepthMap_B,img,MSK;
	private static PXCUPipeline pp;
	private static TestSetMaker testSets ;
	private static Classifier classifier;
	private static Converter cvt;
	
	private static CvSVM[] SVMs;
	
	
	//captureBox location
	private static int x=10,y=10;
	private static int ClosestX=0,ClosestY=0;
	private static int SecondX=0,SecondY=0;
	
	private static double MIN_DIST_VAL=0;

	private static IplImage DepthImg;
	
	public static void init()
	{
		
		SVMs= new CvSVM[SVMSIZE];
		
	for (int i = 0; i <SVMSIZE; i++) {
			
			Classifier svm = new Classifier();
			svm.getSVM().load("SVM_TRAINED"+i, "_0218");
		SVMs[i]=svm.getSVM();
		}
	
		cvt = new Converter();
		dataSets = new CvMat[2];
		dataTest = cvCreateMat(1, DATSIZE, CV_32FC1);
		
		for (int j = 0; j < 2; j++) {

			dataSets[j] = cvCreateMat(testNum, DATSIZE, CV_32FC1);

		}
		
		
		classifier= new Classifier();
		
		
		
		cvRect(CaputureBox_X, CaputureBox_Y, CaputureBox_WIDTH, CaputureBox_HEIGHT);
		cvRect(0, 0, CaputureBox_WIDTH, CaputureBox_HEIGHT);
		
		captureArr= new short[CaputureBox_MAT_SIZE];
		depthmap_V= new short[MATSIZE];
		depthmap_R= new short[MATSIZE];
		depthmap_G= new short[MATSIZE];
		depthmap_B= new short[MATSIZE];
		Size= new int[2];
		DepthImg = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		DepthMap = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		MSK = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		DepthMap_R = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		DepthMap_G = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		DepthMap_B = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		
		testMap = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
		capture = cvCreateImage(cvSize(CaputureBox_WIDTH, CaputureBox_HEIGHT), IPL_DEPTH_8U, 1);
		DepthMap_3C = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U,3);
		img = cvCreateImage(cvSize(CaputureBox_WIDTH, CaputureBox_HEIGHT), IPL_DEPTH_8U,1);
		
		cvSetZero(DepthMap);
		cvSetZero(testMap);
		cvSetZero(capture);
		
		pp=new PXCUPipeline();
		 
        if (!pp.Init(PXCUPipeline.GESTURE)) {

	            System.out.print("Failed to initialize PXCUPipeline\n");

	            System.exit(3);
	        }
	}
	public static void testSet_init()
	{
		new TestSetMaker();
		
		testSets = new TestSetMaker();
		testSets.createTestFile("hand4.dat");
		testSets.recordReady(90, 90	);
	}
	
	public static void makeTestSet(TestSetMaker ts,String file)
	{
		ts.createTestFile(file);
		ts.recordReady(320,240);
		
		   for (int k =0;k<50;k++) {

	            if (!pp.AcquireFrame(true)) break;

	           if(! pp.QueryDepthMap(depthmap_V))break;
	           
	           pp.QueryDepthMapSize(Size);
	           
	         
	           
	           for (int i = 0; i < depthmap_V.length; i++) {
				if((depthmap_V[i]<MAX_DIST))
					depthmap_V[i]=(short) ((depthmap_V[i]/(double)MAX_DIST)*255);
	
			}
	           
	            ts.recordTestFrameSet(depthmap_V,320,240);
	            
	            DepthMap=cvt.CvtArr2Img(DepthMap, depthmap_V, 320, 240);
	            cvNot(DepthMap, DepthMap);	          
		        cvShowImage("depth", DepthMap);
		      
		        pp.ReleaseFrame();
		         cvWaitKey(10);
		   }
		ts.recordFinish();
		   
	}
	
	
	public static void loadTestSet(String fname, int width,int height,int classnum)
	{
		TestSetMaker ts1= new TestSetMaker();
		TestSetMaker ts2= new TestSetMaker();
		ts1.loadReady(fname,width);
		
		IplImage result = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		short[] depthData = new short[width*height];
	
		 FeatureDescriptor fd0 = new FeatureDescriptor();
			
		CvFont font = new CvFont();
		CvMat mat ;
		Pointer pt;
		CvMat data0;
		cvInitFont(font, CV_FONT_HERSHEY_COMPLEX, 0.5, 0.5,0,0,0	);
		IplImage datasets = cvCreateImage(cvSize(DATSIZE, 100), IPL_DEPTH_8U, 1);
		
		int n=0;
		
	
	for(;;){
		if(n<testNum/2) 
      	 {
			depthData=ts1.loadTestFrameSet();
			
       		depthData=Smoothing(img,depthData, CaputureBox_WIDTH, CaputureBox_HEIGHT);
			
      	    data0=fd0.MK(depthData, 0, 0, 90, 90);
      	  
      	 
      	
      	
      		for (int j = 0; j < DATSIZE; j++) {
      			
				dataSets[0].put(n, j, (float)data0.get(0, j));
				if(!(dataSets[0].get(n, j)>=0)){
					dataSets[0].put(n, j, 0);
				}
				
      		}
      	 }
		else if(n>=testNum/2  && n<testNum){
			if(n==testNum/2)
			{
				ts2.loadReady("negative"+classnum%2+".dat",width);
			}
			
			depthData=ts2.loadTestFrameSet();
			
       		depthData=Smoothing(img,depthData, CaputureBox_WIDTH, CaputureBox_HEIGHT);
			
     	    data0=fd0.MK(depthData, 0, 0, 90, 90);
    
      		for (int j = 0; j < DATSIZE; j++) {
      			
				dataSets[0].put(n, j, (float)data0.get(0, j));
				if(!(dataSets[0].get(n, j)>=0)){
					
					dataSets[0].put(n, j, 0);
				}
				
      		}
			
			
			}
      	else if (n==testNum){
      	 classifier.trainSVM(dataSets[0], testNum,classnum);
      	 
      	 }
      	 
      	 if(n==testNum+1){
      		
      		 int prob=0;
      		 
      		 for (int i = 0; i < testNum; i++) {
      			for (int j = 0; j < DATSIZE; j++)
    				dataTest.put(0, j, (float)dataSets[0].get(i, j));
      			
      			if(i<testNum/2)
      			if(classifier.getSVM().predict(dataTest, true)==1)
      				prob++;
      			System.out.println(i+" : "+classifier.getSVM().predict(dataTest, false));		
			}
      		 
      		 
      		System.out.println("TT Rate=" +(float)prob/testNum);
      		System.out.println("TF Rate=" +(1-(float)prob/testNum));
      		
      		
      		classifier.getSVM().save("SVM_TRAINED"+classnum, "_0218");
    
      	
      	 }
      	 if(n>testNum+1)
      	 {
      		 cvShowImage("datasets", dataSets[0]);
      		 break;
      	 }
      	 
	          result= cvt.CvtArr2Img(img,depthData, width, height);
        	  cvNot(result, result);
        	  cvPutText(result, "Cnt:"+n++ , cvPoint(10, 	20), font, CV_RGB(0,0,255));	
        	  cvShowImage("depth", result);
        	  cvWaitKey(1);
	    

	           //cvReleaseMat(data0);
	          
	     
		}
		
	}
	public static void loadTestSets(int classnum)
	{
//		TestSetMaker ts1= new TestSetMaker();
//		TestSetMaker ts2= new TestSetMaker();
//		
//		short[] depthData1 = new short[90*90];
//		short[] depthData2 = new short[90*90];
//		
//		FeatureDescriptor fd0 = new FeatureDescriptor();
//      	FeatureDescriptor fd1 = new FeatureDescriptor();
//    	    
//		ts1.loadReady("hand1"+".dat",90);
//		ts2.loadReady("hand2"+".dat",90);
//		
//			System.out.println("training");
//		CvMat data0 = null,data1 = null;
//		for(int i =0; i<testNum;i++){
//			
//			depthData1=ts1.loadTestFrameSet();
//					depthData2=ts2.loadTestFrameSet();
//
//			img=cvt.CvtArr2Img(img,depthData1,CaputureBox_WIDTH,CaputureBox_HEIGHT);
//			
//			cvSmooth(img, img, CV_GAUSSIAN, 3);
//			depthData1= cvt.CvtImg2Arr(img);
//			
//				
//      	  	data0=fd0.get1DHistogram(depthData1, 0, 0);
//      	  	
//      	  	img=cvt.CvtArr2Img(img,depthData2,CaputureBox_WIDTH,CaputureBox_HEIGHT);
//			
//			cvSmooth(img, img, CV_GAUSSIAN, 3);
//			depthData2= cvt.CvtImg2Arr(img);
//			
//			
//      	  	
//      		data1=fd1.get1DHistogram(depthData2, 0, 0);
//    
//    	
//    	
//      		for (int j = 0; j < DATSIZE; j++) {
//				dataSets[0].put(0, j, (float)data0.get(0, j));
//				if(!(dataSets[0].get(0, j)>=0))
//					dataSets[0].put(0, j, 0);
//				
//      	  		dataSets[0].put(1, j, (float)data1.get(0, j));
//      	  	if(!(dataSets[0].get(1, j)>=0))
//				dataSets[0].put(1, j, 0);
//      	  }
//      		if(i==1)
//      		for (int j = 0; j < DATSIZE; j++){
//      			
//				dataTest.put(0, j, (float)data0.get(0, j));
//				if(!(dataTest.get(0, j)>=0))
//				{
//					dataTest.put(0, j,0);
//				}
//      		}
//      		
//      System.out.println(i);
//		classifier.trainSVM(dataSets[0], 2,classnum);
//		
//		
//		
//		}
//
//		classifier.classifySVM(dataTest);
//		
//		ts1.loadFinish();
//		ts1.loadReady("hand1"+".dat",90);
//		for (int i = 0; i < 90; i++) {
//			depthData1=ts1.loadTestFrameSet();
//			
//			img=cvt.CvtArr2Img(img,depthData1,CaputureBox_WIDTH,CaputureBox_HEIGHT);
//			
//			cvSmooth(img, img, CV_GAUSSIAN, 3);
//			depthData1= cvt.CvtImg2Arr(img);
//			
//			fd0.MK(depthData1, 0, 0, 90, 90);
//			
//			
//			for (int j = 0; j < DATSIZE; j++) {
//				dataTest.put(0,j , (float)data0.get(0, j));
//				if(!(dataTest.get(0, j)>=0))
//				{
//					dataTest.put(0, j,0);
//				}
//			}
//			System.out.println(classifier.getSVM().predict(dataTest, false));
//		}
//		
//		ts1.loadFinish();
//		
//		
//		
//		
//		
//		cvReleaseMat(data0);
//		cvReleaseMat(data1);
	}
	
	
	public static int matchHand(CvMat data)
	{
		MIN_DIST_VAL=1000;
		int index=-1;		
		float minDist=100;
		float dist=0;
		
		for (int i = 0; i < SVMSIZE; i++) {
			//System.out.println(i+" svm:"+SVMs[i].predict(data, true));
			if((dist=SVMs[i].predict(data, true))<minDist)
				{
					cnt++;
					minDist=dist;
					index=i;
				}			
		}
		
		if(minDist>=0){
			index=-1;
			//System.out.println(cnt+" duplicate hand");
		}
		else
		MIN_DIST_VAL=minDist;
		
		return index;
	}
	public static short[] Smoothing(IplImage img,short[] depthData,int width, int height)
	{
		img=cvt.CvtArr2Img(img,depthData,width,height);
		
		cvSmooth(img, img, CV_GAUSSIAN, 3);
		
		return cvt.CvtImg2Arr(img);
	}
	public static void drawCaputeBox(int x, int y)
	{
		
		cvDrawCircle(DepthMap_3C,cvPoint(ClosestX,ClosestY),3 , cvScalar(255,255, 255, 0), 2,1, 0);
        
		x=x<0?0:x;
		x=x>319-CaputureBox_WIDTH?319-CaputureBox_WIDTH:x;
		y=y<0?0:y;
		y=y>239-CaputureBox_HEIGHT?239-CaputureBox_HEIGHT:y;
		
		   	  cvDrawRect(DepthMap_3C,cvPoint(x, y),cvPoint(90+x,90+y), cvScalar(0,255, 0, 0), 2,0, 0);
	          cvDrawCircle(DepthMap_3C,cvPoint(45+x, 45+y),2 , cvScalar(0,255, 0, 0), 2,0, 0);
	          
	          cvDrawCircle(DepthMap_3C,cvPoint(45+x, 45+y),36, cvScalar(0,255, 0, 0), 1,0, 0);
	          cvDrawCircle(DepthMap_3C,cvPoint(45+x, 45+y),27, cvScalar(0,255, 0, 0), 1,0, 0);
	          cvDrawCircle(DepthMap_3C,cvPoint(45+x, 45+y),18, cvScalar(0,255, 0, 0), 1,0, 0);
	          cvDrawCircle(DepthMap_3C,cvPoint(45+x, 45+y), 9, cvScalar(0,255, 0, 0), 1,0, 0);
	          for (int i = 0; i < 8; i++) {
				
	        	  cvLine(DepthMap_3C, cvPoint(45+x,45+y), cvPoint((int)(36*Math.cos((i*45/180.0)*Math.PI))+45+x,(int)(36*Math.sin((i*45/180.0)*Math.PI))+45+y), cvScalar(0,255, 0, 0),1, 0, 0);
			}
	          
	}
	
	public static void slidingWindow(int ClosestX, int ClosestY	,FeatureDescriptor fd)
	{
		 boolean found =false;
    	 short[] hand = new short[90*90]; 
    	 HandFinder hf = new HandFinder();
    	 CvMat handDat = null;
   

    	int startX =(ClosestX-90)<0?0:ClosestX-90;
    	int startY =(ClosestY-90)<0?0:ClosestY-90;
  
    	int handNum=-1;
    	double distVal=1000;
    	int finalHandNum=-1;
    	
    	
        depthmap_V=Smoothing(DepthMap, depthmap_V, 320, 240);
    	fd.MK(depthmap_V,ClosestX,ClosestY,320,240);
    	
    	for (int i = startX; i < ClosestX; i+=15) {
    		for (int j = startY; j < ClosestY; j+=15) {
				
//    			hand=hf.FindHandSW(DepthMap, i, j,depthmap_V);
//    			
//    			cvShowImage("handFinding", hf.getImg());
//    		    handDat=fd.get1DHistogram(hand, 0, 1);
//    		    dataTest=correctNAN(handDat, dataTest);
//    		    cvWaitKey(1);

    			
//    			  caputureBoxImage(DepthMap,capture,i,j,0);
//	        	
//	  			  captureArr= Smoothing(img,captureArr, CaputureBox_WIDTH, CaputureBox_HEIGHT);
	  				
	  			  
	        	//  handDat=fd.get1DHistogram(captureArr, 0, 0);
	        	  
    			
	        	 // dataTest=correctNAN(handDat, dataTest);
	        	 
	        	 // cvWaitKey(1);
	        	  
    		if((handNum=matchHand(handDat))!=-1){
    			
    			if(distVal>MIN_DIST_VAL){
    				distVal=MIN_DIST_VAL;
    			finalHandNum=handNum;
    			x=i;
    			y=j;
    			//cvShowImage("handFound", handDat);
    			}
    		}
    		cvShowImage("handFound", handDat);
		}
		
    	}
    	System.out.println(finalHandNum+":"+x+","+y);	
			cvReleaseMat(handDat);	
		
//    	if(found!=true)
//    		System.out.println("can't find");
    	
	}
	 
	public static CvMat correctNAN(CvMat data,CvMat dataTest)
	{
		 for (int j = 0; j < DATSIZE; j++) {
 			
				dataTest.put(0, j, (float)data.get(0, j));
				
				if(!(dataTest.get(0, j)>=0)){
					
					dataTest.put(0, j, 0);
				}
				
				}
     	  return dataTest;
	}
	
	
	public static short[] findClosestArea(short[] map,short[] depthMap,int mode)
	{
		int ClosestValue= 1000;
		int ClosestIdx=-1;
		
			
		for (int i = 0; i < depthMap.length; i++) {
		
			if(mode==0){
				if(depthMap[i]<ClosestValue){
					ClosestValue=depthMap[i];
					ClosestIdx=i;
				}
			}
			else{
				if(!(i%320>=ClosestX-45 && i%320<ClosestX-45+CaputureBox_WIDTH && i/320>=ClosestY-45 && i/320<ClosestY-45+CaputureBox_HEIGHT  ))	
				{//System.out.println(ClosestValue+","+depthMap[i]);
					if(depthMap[i]<ClosestValue){
						ClosestValue=depthMap[i];
						ClosestIdx=i;
						
					}
				}
			
			}
			
		}
		
		if(mode==0){
		if(Math.abs(depthMap[ClosestX+ClosestY*320]-depthMap[ClosestIdx%320+ClosestIdx/320*320])>10)
		{
			ClosestX= ClosestIdx%320;
			ClosestY= ClosestIdx/320;
		
		}
		}
		else
		{						
				SecondX= ClosestIdx%320;
				SecondY= ClosestIdx/320;				
				
		}

	
		int min=ClosestValue;
		int max=min+(int) ((130/(double)MAX_DIST)*255);
		
		
	
		
		for (int i = 0; i < depthMap.length; i++) {
			int X=i%320;
			int Y=i/320;
			double r=0;
			if(mode==0)
				r= Math.sqrt(Math.pow(X-ClosestX,2)+Math.pow(Y-ClosestY,2));
			else
				r= Math.sqrt(Math.pow(X-SecondX,2)+Math.pow(Y-SecondY,2));
				
				
				if(((depthMap[i]<max&&depthMap[i]>min)&&(r<95)))
					{
						short v=depthMap[i];
						map[i]=v;
					}
				else
					map[i]=255;
		}
		
			
		return map;
		
	}
	public static void MEANShift(short[] depthMap,int originX, int originY,int mode)
	{
		double d=1000;
		int Xsum=0;
		int Ysum=0;
		int n=1;
		
		int meanX=0;
		int meanY=0;
		
		
		while(!(d<5))	
		{
			for (int i = 0; i < depthMap.length; i++) {
				
				if(depthMap[i]!=255)
				{
					n++;
					Xsum+=i%320;
					Ysum+=i/320;
					
				}
			}
			meanX=Xsum/n;
			meanY=Ysum/n;
			
			d=Math.sqrt(Math.pow(meanX-originX, 2)+Math.pow(meanY-originY, 2));
			originX=meanX;
			originY=meanY;
			Xsum=0;
			Ysum=0;
			n=1;
			//System.out.println(d);
			
		}
		if(mode==0){
		ClosestX=meanX;
		ClosestY=meanY;
		}
		else
		{
			SecondX=meanX;
			SecondY=meanY;
		}
     	
	}
	
	public static CvSeq findBiggestContour(CvSeq contours)
	{
		CvSeq MaxContourPtr = null;
		CvSeq MaxContourPtr1 = null;
		CvRect contourBox = null;
		CvRect contourBox1 = null;
		int boxArea=0;
		int maxArea=-1;
		
		
		for (CvSeq ptr = contours; ptr != null; ptr = ptr.h_next()) {

			contourBox = cvBoundingRect(ptr, 1);
			
			boxArea=contourBox.width()*contourBox.height();
			
			if(boxArea<45*45 //|| boxArea>100*100
					)
				continue;
			
			if(boxArea>maxArea){
				
				//cvDrawRect(DepthMap_3C, cvPoint(contourBox.x(), contourBox.y()),cvPoint(contourBox.x()+contourBox.width(), contourBox.y()+contourBox.height()), cvScalar(0,255, 0, 0), 2,0, 0);
				
				maxArea=boxArea;
				MaxContourPtr=ptr;
			}
			
		}
		
		
	
		
//	if(contourBox!=null)
//		cvDrawRect(DepthMap_3C, cvPoint(contourBox.x(), contourBox.y()),cvPoint(contourBox.x()+contourBox.width(), contourBox.y()+contourBox.height()), cvScalar(0,255, 0, 0), 2,0, 0);
//		
	
	
	return MaxContourPtr;
				
			
		
	}
	public static void findHand(IplImage DepthImg)
	{
		cvThreshold(DepthImg, MSK, 0, 255, CV_THRESH_BINARY);
		
		
		CvMemStorage mem = cvCreateMemStorage(0);
		CvSeq contours = new CvSeq();
		CvSeq Maxcontour;
		
		cvShowImage("MSK", MSK	);  
		int contourNum = cvFindContours(MSK, mem, contours,
				Loader.sizeof(CvContour.class), CV_RETR_LIST,
				CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
		
		if (contourNum > 0)
		{
			Maxcontour=findBiggestContour(contours);
			
			if(Maxcontour!=null){
			
				CvMemStorage storage = cvCreateMemStorage(0);
				
				CvSeq poly = cvApproxPoly(Maxcontour, Loader.sizeof(CvContour.class), storage,
						CV_POLY_APPROX_DP, 3, 1);


				cvDrawContours(DepthMap_3C,poly, cvScalar(255, 0, 0, 0), cvScalar(0, 0, 0, 0), 1, 3, 8);			
				
				cvReleaseMemStorage(storage);
			}
		
		}
		cvReleaseMemStorage(mem);
		

		
	}
	public static short[] maskDepthMap(short[] depthmap,int centerX,int centerY)
	{
		int minA=depthmap[ClosestX+ClosestY*320];
		int maxA=minA+(int) ((150/(double)MAX_DIST)*255);
		
		int minB=depthmap[SecondX+SecondY*320];
		int maxB=minB+(int) ((130/(double)MAX_DIST)*255);
		
	
		for (int i = 0; i < depthmap.length; i++) {
			boolean ishand1
					= i%320>=centerX&& i%320<centerX+CaputureBox_WIDTH && i/320>=centerY && i/320<centerY + CaputureBox_HEIGHT
					  //&&(depthmap[i]<maxA && depthmap[i]>minA)
					  ;
//			boolean ishand2
//					= i%320>=SecondX-CaputureBox_WIDTH/2 && i%320<SecondX-CaputureBox_WIDTH/2+CaputureBox_WIDTH && i/320>=SecondY-CaputureBox_HEIGHT/2 && i/320<SecondY-CaputureBox_HEIGHT/2 + CaputureBox_HEIGHT
//							//&&(depthmap[i]<maxB&&depthmap[i]>minB)
//							;	

			if(!(ishand1)){
				depthmap[i]=255;
				
			}
		
		}
		return depthmap;
	}
	
	
	public static void realTimeShow()
	{
		CvCapture cap1 = cvCreateCameraCapture(0);
		
		 FeatureDescriptor fd = new FeatureDescriptor();
         
		boolean Chkmatch=false;
		int cnt=0;
		
		CvFont font= new CvFont();
  	  cvInitFont(font, CV_FONT_HERSHEY_COMPLEX, 0.7, 0.7,0,0,0	);
		  for (;;) {
			  RgbMap = cvQueryFrame(cap1);
	            if (!pp.AcquireFrame(true)) break;
	     
	            pp.QueryDepthMapSize(Size);
          
	           if(! pp.QueryDepthMap(depthmap_V))break;
	    
	          
	           
	           for (int i = 0; i < depthmap_V.length; i++) {

	        	   	depthmap_B[i]=0;
					depthmap_G[i]=0;
					depthmap_R[i]=0;
					
				if((depthmap_V[i]<MAX_DIST))
					{
					depthmap_V[i]=(short) Math.round(((depthmap_V[i]/(double)MAX_DIST)*255));
					
					if(depthmap_V[i]<130)
					depthmap_R[i]=(short) ((depthmap_V[i]/(double)130)*255);
					
					else if(130<depthmap_V[i]&&depthmap_V[i]<200)
						depthmap_G[i]=(short) (depthmap_V[i]/(double)200*255);
					else if(200<depthmap_V[i]&&depthmap_V[i]<255)
						depthmap_B[i]=(short) (depthmap_V[i]/(double)255*255);
	
					}
				else
					depthmap_V[i]=255;
			}
	            DepthMap_R= cvt.CvtArr2Img(DepthMap_R,depthmap_R,320,240);
          		DepthMap_G= cvt.CvtArr2Img(DepthMap_G,depthmap_G,320,240);
          		DepthMap_B= cvt.CvtArr2Img(DepthMap_B,depthmap_B,320,240);
          		DepthMap= cvt.CvtArr2Img(DepthMap,depthmap_V,320,240);
          		
          		
          		cvNot(DepthMap, DepthMap);	
	            cvMerge(DepthMap_B, DepthMap_G,DepthMap_R, null, DepthMap_3C);
	     
	            
	          
		            short[] map= new short[MATSIZE];
		            map= findClosestArea(map,depthmap_V,0);
		            MEANShift(map, ClosestX, ClosestY,0);
		            drawCaputeBox(ClosestX-45, ClosestY-55);
		    
//		            short[] map1= new short[MATSIZE];
//		            if((map1= findClosestArea(map1,depthmap_V,1))!=null)
//		            	{	
//		            		MEANShift(map1, SecondX, SecondY,1);
//		            		 drawCaputeBox(SecondX-45, SecondY-45);
//		            	}
		            
//		            map=maskDepthMap(map,ClosestX-45, ClosestY-55);
//		            DepthImg=cvt.CvtArr2Img(DepthImg,map, 320, 240);
//		            cvNot(DepthImg, DepthImg);
		           // findHand(DepthImg);
		          
		        
		        
		          if(Chkmatch==true){
		        	  caputureBoxImage(DepthMap,capture,ClosestX-45,ClosestY-55,0);
		        	  captureArr=Smoothing(img, captureArr, 90, 90);
		          
		        	  dataTest=correctNAN(fd.MK(captureArr, 0, 0, 90, 90), dataTest);
		        	  
		        	  int i=fd.getDominanAngleIdx();
		        	  //System.out.println((360-45*i)+"~"+(360-45*(i+1)));
		        	  
		        	  cvLine(DepthMap_3C, cvPoint(ClosestX,ClosestY-10), cvPoint((int)(36*Math.cos((i*45/180.0)*Math.PI))+ClosestX,(int)(36*Math.sin((i*45/180.0)*Math.PI))+ClosestY-10), cvScalar(255,0, 0, 0),3, 0, 0);

		        	 int handNum=matchHand(dataTest);
		        	 
		        	 if(handNum!=-1)
		        	  cvPutText(DepthMap_3C, handNum+"" , cvPoint(45, 45), font, CV_RGB(255,0,0));
		        	 
		          }
		          switch(cvWaitKey(1))
		          {
		          
		          case 'c':
		        	  
		        	 Chkmatch=(Chkmatch==true)?false:true;
//		        	  System.out.println(cnt++);
		        	 
		        	  break;
		          case 's':
		        	  testSets.recordFinish();
		        	  System.out.println("record finish");
		        	  System.exit(1);
		        	  break;
		       
//		          case 'n':
//		        	  
//		        	  caputureBoxImage(DepthMap,capture,x,y,0);
//		        	  FeatureDescriptor fd3 = new FeatureDescriptor();
//		        	  
//		  			  captureArr= Smoothing(img,captureArr, CaputureBox_WIDTH, CaputureBox_HEIGHT);
//		  				
//		        	  CvMat data3=fd3.get1DHistogram(captureArr, 0, 1);
//		        	  
//		        	  dataTest=correctNAN(data3, dataTest);
//		        	 
//		        	  System.out.println(matchHand(dataTest));
//		        	
//		        	  break;
		         
		          case 'f': 
		        	 
		        	 slidingWindow(ClosestX,ClosestY,fd);
		        	  
		        	  break;
		        	  
		          }
		          
		          cvShowImage("depth3C", DepthMap_3C);
		          cvShowImage("depth1C", DepthMap);
			      cvShowImage("RGB", RgbMap	);   
			     
				     
			     
			      
			      
		          pp.ReleaseFrame();
		   }
		  cvReleaseImage(DepthMap);
	      cvReleaseImage(DepthMap_3C);
	      cvReleaseImage(RgbMap);
	      cvReleaseImage(MSK);
	      
	}
	public static void caputureBoxImage(IplImage DepthMap,IplImage capture,int x,int y,int mode)
	{
		
		x=x<0?0:x;
		x=x>319-CaputureBox_WIDTH?319-CaputureBox_WIDTH:x;
		y=y<0?0:y;
		y=y>239-CaputureBox_HEIGHT?239-CaputureBox_HEIGHT:y;
		
		cvRect(x, y, CaputureBox_WIDTH, CaputureBox_HEIGHT);
		
		cvSetImageROI(DepthMap,  cvRect(x, y, CaputureBox_WIDTH, CaputureBox_HEIGHT));
		cvCopy(DepthMap, capture);
		cvResetImageROI(DepthMap);
		
		
		for(int i=0,j=0; i< MATSIZE;i++)
		{
			if((i%320)>=x &&(i%320)<CaputureBox_WIDTH+x && (i/320)>=y &&(i/320)<CaputureBox_HEIGHT+y)
			{
				captureArr[j++]=depthmap_V[i];
			}
		}
		cvShowImage("CaptureBox",capture);
		/// save Frame continuously
		if(mode==1)
		testSets.recordTestFrameSet(captureArr, CaputureBox_WIDTH, CaputureBox_HEIGHT);
		
      
	}
	

	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		
        	init();
        
        	//testSet_init();
        	// makeTestSet();
        	//loadTestSet("hand1.dat",90,90,0);
        	//loadTestSet("hand2.dat",90,90,1);
        	//loadTestSet("hand3.dat",90,90,2);
        	//loadTestSet("hand4.dat",90,90,3);
        	
        	//loadTestSets(2);
        	//loadTestSets();
        	//loadTestSets();
        	//loadTestSets();
        	
        	realTimeShow();
        	
        
        	
        	pp.Close();

	        System.exit(0);

	}

}
