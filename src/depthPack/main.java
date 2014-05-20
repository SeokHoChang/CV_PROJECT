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
import com.googlecode.javacv.cpp.opencv_core.CvFont;
import com.googlecode.javacv.cpp.opencv_core.CvMat;
import com.googlecode.javacv.cpp.opencv_core.CvMemStorage;
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

import intel.pcsdk.*;

public class main {

	private static final int SVMSIZE = 4;
	
	private static final int MATSIZE = 320*240;
	
	private static final int DATSIZE = 576;
	
	private static final int testNum= 180;
	///Used measure is Milimeter
	private static final int MAX_DIST = 1000;//mm
	private static final int MIN_SEARCH_RANGE=270;//mm
	private static final int MAX_SEARCH_RANGE=500;//mm
	private static final int RANGE=50;//mm
	 
	private static final int CaputureBox_X=10;
	private static final int CaputureBox_Y=10;
	private static final int CaputureBox_WIDTH=90;
	private static final int CaputureBox_HEIGHT=90;
	private static final int CaputureBox_MAT_SIZE=90*90;

	public static int cnt=0;
	private static final double THRESHOLD= 0.4;
	private static final double RATIO = 0.5; 
	private static  HashMap<Number, Number> map ;
	
	private static CvMat dataTest;
	private static CvMat data;
	private static CvMat[] dataSets;
	
	private static short[] depthmap_V,depthmap_R,depthmap_G,depthmap_B,depthmap_temp;

	private static short[] captureArr;
	private static int[] Size,RGB,RGB_SIZE;
	private static IplImage RgbMap,DepthMap,testMap,capture,DepthMap_3C,CurrentROI,DepthMap_R,DepthMap_G,DepthMap_B,img;
	private static int minRange,maxRange,Range;
	private static PXCUPipeline pp;
	private static TestSetMaker tsMkr,testSets ;
	private static Classifier classifier;
	private static CvRect CaptureBox,ROIBox;
	
	private static CvMemStorage storage,storage1;
	private static CvSeq objectKeypoints,objectDescriptors, imageKeypoints, imageDescriptors ;
	private static CvSURFParams params;
	private static Converter cvt;
	
	private static CvSVM[] SVMs;
	
	
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
		
		
		
		CaptureBox = cvRect(CaputureBox_X, CaputureBox_Y, CaputureBox_WIDTH, CaputureBox_HEIGHT);
		ROIBox = cvRect(0, 0, CaputureBox_WIDTH, CaputureBox_HEIGHT);
		
		captureArr= new short[CaputureBox_MAT_SIZE];
		depthmap_V= new short[MATSIZE];
		depthmap_R= new short[MATSIZE];
		depthmap_G= new short[MATSIZE];
		depthmap_B= new short[MATSIZE];
		depthmap_temp= new short[MATSIZE];
		
		RGB 	  = new int[10000000];
		Size= new int[2];
		RGB_SIZE= new int[2];
		
		CurrentROI = cvCreateImage(cvSize(CaputureBox_WIDTH, CaputureBox_HEIGHT),IPL_DEPTH_8U,1);
		DepthMap = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 1);
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
		
		minRange=(int) ((MIN_SEARCH_RANGE/(double)MAX_DIST)*255);
        maxRange=(int) ((MAX_SEARCH_RANGE/(double)MAX_DIST)*255);
        Range= (int) ((RANGE/(double)MAX_DIST)*255);
        
        
     
		pp=new PXCUPipeline();
		 
        if (!pp.Init(PXCUPipeline.GESTURE)) {

	            System.out.print("Failed to initialize PXCUPipeline\n");

	            System.exit(3);

	        }
	}
	public static void testSet_init()
	{
		tsMkr= new TestSetMaker();
		
		testSets = new TestSetMaker();
		testSets.createTestFile("hand3.dat");
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
	           	  Pointer pt= new ShortPointer(depthmap_V);
		          CvMat mat = cvMat(240, 320, CV_16UC1, pt);
		          cvConvert(mat, DepthMap);
		   
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
			
      	    data0=fd0.get1DHistogram(depthData, 0, 1);
      	  
      	 
      	
      	
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
			
			data0=fd0.get1DHistogram(depthData, 0, 1);
    
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
		TestSetMaker ts1= new TestSetMaker();
		TestSetMaker ts2= new TestSetMaker();
		
		short[] depthData1 = new short[90*90];
		short[] depthData2 = new short[90*90];
		
		FeatureDescriptor fd0 = new FeatureDescriptor();
      	FeatureDescriptor fd1 = new FeatureDescriptor();
    	    
		ts1.loadReady("hand1"+".dat",90);
		ts2.loadReady("hand2"+".dat",90);
		
			System.out.println("training");
		CvMat data0 = null,data1 = null;
		for(int i =0; i<testNum;i++){
			
			depthData1=ts1.loadTestFrameSet();
					depthData2=ts2.loadTestFrameSet();

			img=cvt.CvtArr2Img(img,depthData1,CaputureBox_WIDTH,CaputureBox_HEIGHT);
			
			cvSmooth(img, img, CV_GAUSSIAN, 3);
			depthData1= cvt.CvtImg2Arr(img);
			
				
      	  	data0=fd0.get1DHistogram(depthData1, 0, 0);
      	  	
      	  	img=cvt.CvtArr2Img(img,depthData2,CaputureBox_WIDTH,CaputureBox_HEIGHT);
			
			cvSmooth(img, img, CV_GAUSSIAN, 3);
			depthData2= cvt.CvtImg2Arr(img);
			
			
      	  	
      		data1=fd1.get1DHistogram(depthData2, 0, 0);
    
    	
    	
      		for (int j = 0; j < DATSIZE; j++) {
				dataSets[0].put(0, j, (float)data0.get(0, j));
				if(!(dataSets[0].get(0, j)>=0))
					dataSets[0].put(0, j, 0);
				
      	  		dataSets[0].put(1, j, (float)data1.get(0, j));
      	  	if(!(dataSets[0].get(1, j)>=0))
				dataSets[0].put(1, j, 0);
      	  }
      		if(i==1)
      		for (int j = 0; j < DATSIZE; j++){
      			
				dataTest.put(0, j, (float)data0.get(0, j));
				if(!(dataTest.get(0, j)>=0))
				{
					dataTest.put(0, j,0);
				}
      		}
      		
      System.out.println(i);
		classifier.trainSVM(dataSets[0], 2,classnum);
		
		
		
		}

		classifier.classifySVM(dataTest);
		
		ts1.loadFinish();
		ts1.loadReady("hand1"+".dat",90);
		for (int i = 0; i < 90; i++) {
			depthData1=ts1.loadTestFrameSet();
			
			img=cvt.CvtArr2Img(img,depthData1,CaputureBox_WIDTH,CaputureBox_HEIGHT);
			
			cvSmooth(img, img, CV_GAUSSIAN, 3);
			depthData1= cvt.CvtImg2Arr(img);
			data0=fd0.get1DHistogram(depthData1, 0, 0);
			
			for (int j = 0; j < DATSIZE; j++) {
				dataTest.put(0,j , (float)data0.get(0, j));
				if(!(dataTest.get(0, j)>=0))
				{
					dataTest.put(0, j,0);
				}
			}
			System.out.println(classifier.getSVM().predict(dataTest, false));
		}
		
		ts1.loadFinish();
		
		
		
		
		
		cvReleaseMat(data0);
		cvReleaseMat(data1);
	}
	
	
	public static int matchHand(CvMat data)
	{
		int index=-1;
			
		int cnt=0;
		
		
		for (int i = 0; i < SVMSIZE; i++) {
			
			if(SVMs[i].predict(data, false)!=-1)
				{
					cnt++;
					index=i;
				}
			
		}
		if(cnt!=1 || cnt ==0){
			index=-1;
			//System.out.println(cnt+" duplicate hand");
		}
		
		return index;
	}
	public static short[] Smoothing(IplImage img,short[] depthData,int width, int height)
	{
		img=cvt.CvtArr2Img(img,depthData,width,height);
		
		cvSmooth(img, img, CV_GAUSSIAN, 3);
		depthData= cvt.CvtImg2Arr(img);
		return depthData;
	}
	public static void realTimeShow()
	{
		CvCapture cap1 = cvCreateCameraCapture(0);
		
		
		boolean Chkmatch=false;
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
		     
		          if(Chkmatch==true){
		      
		          }
		        
		          cvDrawRect(DepthMap_3C,cvPoint(10, 10),cvPoint(100,100), cvScalar(0,255, 0, 0), 2,0, 0);
		          cvDrawCircle(DepthMap_3C,cvPoint(55, 55),2 , cvScalar(0,255, 0, 0), 2,0, 0);
		          
		          cvDrawCircle(DepthMap_3C,cvPoint(55, 55),36, cvScalar(0,255, 0, 0), 1,0, 0);
		          cvDrawCircle(DepthMap_3C,cvPoint(55, 55),27, cvScalar(0,255, 0, 0), 1,0, 0);
		          cvDrawCircle(DepthMap_3C,cvPoint(55, 55),18, cvScalar(0,255, 0, 0), 1,0, 0);
		          cvDrawCircle(DepthMap_3C,cvPoint(55, 55), 9, cvScalar(0,255, 0, 0), 1,0, 0);
		          for (int i = 0; i < 8; i++) {
					
		        	  cvLine(DepthMap_3C, cvPoint(55,55), cvPoint((int)(36*Math.cos((i*45/180.0)*Math.PI))+55,(int)(36*Math.sin((i*45/180.0)*Math.PI))+55), cvScalar(0,255, 0, 0),1, 0, 0);
				}
		          
	            
		          switch(cvWaitKey(10))
		          {
		          
		          case 'c':
		        	  
		        	  caputureBoxImage(DepthMap,capture,1);
		        	  System.out.println(cnt++);
		        	 
		        	  break;
		          case 's':
		        	  testSets.recordFinish();
		        	  System.out.println("record finish");
		        	  System.exit(1);
		        	  break;
		       
		          case 'n':
		        	  
		        	  caputureBoxImage(DepthMap,capture,0);
		        	  FeatureDescriptor fd3 = new FeatureDescriptor();
		        	  
		        	  
		        	   
		  			   captureArr= Smoothing(img,captureArr, CaputureBox_WIDTH, CaputureBox_HEIGHT);
		  				
		        	  CvMat data3=fd3.get1DHistogram(captureArr, 0, 1);
		        	  
		        	  for (int j = 0; j < DATSIZE; j++) {
		        			
		  				dataTest.put(0, j, (float)data3.get(0, j));
		  				
		  				if(!(dataTest.get(0, j)>=0)){
		  					
		  					dataTest.put(0, j, 0);
		  				}
		  				
		  				}
		        	  
		        	  System.out.println(matchHand(dataTest));
	  	
		        	  break;
		         
		          case 'f': 
	
		        	  break;
		        	  
		          }
		          
		          cvShowImage("depth3C", DepthMap_3C);
		          cvShowImage("depth1C", DepthMap);
			      cvShowImage("RGB", RgbMap	);   
			     
			      
		          pp.ReleaseFrame();
		   }
	}
	public static void caputureBoxImage(IplImage DepthMap,IplImage capture,int mode)
	{
		
		cvSetImageROI(DepthMap, CaptureBox);
		cvCopy(DepthMap, capture);
		cvResetImageROI(DepthMap);
		
		
		
		cvShowImage("CaptureBox",capture);
		
		
		
		for(int i=0,j=0; i< MATSIZE;i++)
		{
			if((i%320)<CaputureBox_WIDTH && (i/320)<CaputureBox_HEIGHT)
			{
				captureArr[j++]=depthmap_V[i];
			}
		}
		
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
