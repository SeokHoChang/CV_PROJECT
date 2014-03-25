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
import intel.pcsdk.*;

public class main {

	private static final int MATSIZE = 320*240;
	
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
	
	
	private static final double THRESHOLD= 0.4;
	private static final double RATIO = 0.5; 
	private static  HashMap<Number, Number> map ;
	
	
	private static CvMat data;
	
	private static short[] depthmap_V,depthmap_R,depthmap_G,depthmap_B,depthmap_temp;

	private static short[] captureArr;
	private static int[] Size,RGB,RGB_SIZE;
	private static IplImage RgbMap,DepthMap,testMap,capture,DepthMap_3C,CurrentROI,DepthMap_R,DepthMap_G,DepthMap_B;
	private static int minRange,maxRange,Range;
	private static PXCUPipeline pp;
	private static TestSetMaker tsMkr ;
	private static Classifier classifier;
	private static CvRect CaptureBox,ROIBox;
	
	private static CvMemStorage storage,storage1;
	private static CvSeq objectKeypoints,objectDescriptors, imageKeypoints, imageDescriptors ;
	private static CvSURFParams params;
	public static void init()
	{
		
		data = cvCreateMat(3, 576, CV_32FC1);
		
		tsMkr= new TestSetMaker();
		TestSetMaker.createTestFile();
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
		
		cvSetZero(DepthMap);
		cvSetZero(testMap);
		cvSetZero(capture);
		
		minRange=(int) ((MIN_SEARCH_RANGE/(double)MAX_DIST)*255);
        maxRange=(int) ((MAX_SEARCH_RANGE/(double)MAX_DIST)*255);
        Range= (int) ((RANGE/(double)MAX_DIST)*255);
        
        
        storage = CvMemStorage.create();
        opencv_core.cvClearMemStorage(storage);
        
        storage1 = CvMemStorage.create();
        opencv_core.cvClearMemStorage(storage1);
   	  	objectKeypoints = new CvSeq();
	 
   	  	objectDescriptors = new CvSeq();
   	  	
   	  	imageKeypoints = new CvSeq();

        imageDescriptors = new CvSeq();

   	  	
   	  	params = cvSURFParams(500, 1); 
     
     
		pp=new PXCUPipeline();
		 
        if (!pp.Init(PXCUPipeline.GESTURE)) {

	            System.out.print("Failed to initialize PXCUPipeline\n");

	            System.exit(3);

	        }
	}
	public static void makeTestSet()
	{
		TestSetMaker.recordReady();
		
		   for (int k =0;k<50;k++) {

	            if (!pp.AcquireFrame(true)) break;

	           if(! pp.QueryDepthMap(depthmap_V))break;
	           
	           pp.QueryDepthMapSize(Size);
	           
	         
	           
	           for (int i = 0; i < depthmap_V.length; i++) {
				if((depthmap_V[i]<MAX_DIST))
					depthmap_V[i]=(short) ((depthmap_V[i]/(double)MAX_DIST)*255);
	
			}
	           
	            TestSetMaker.recordTestFrameSet(depthmap_V);
	           	  Pointer pt= new ShortPointer(depthmap_V);
		          CvMat mat = cvMat(240, 320, CV_16UC1, pt);
		          cvConvert(mat, DepthMap);
		   
		          cvNot(DepthMap, DepthMap);	          
		          cvShowImage("depth", DepthMap);
		        
	       
		            pp.ReleaseFrame();
		           cvWaitKey(10);
		   }
		TestSetMaker.recordFinish();
		   
	}
	
	
	public static void loadTestSet()
	{
		TestSetMaker.loadReady();
		
		for(;;){
			if((depthmap_V=TestSetMaker.loadTestFrameSet())==null)
				{
					TestSetMaker.loadFinish();
					return;
				}
			
			 Pointer pt= new ShortPointer(depthmap_V);
	          CvMat mat = cvMat(240, 320, CV_16UC1, pt);
	          cvConvert(mat, DepthMap);
	          
	         
	          
	          for(int i=minRange;i<maxRange;i++){
	        	  cvInRangeS(DepthMap, cvScalarAll(i), cvScalarAll(i+Range), testMap);
	        	 // cvNot(testMap, testMap);
	        	 // cvShowImage("test", testMap);
	        	  //templateMatch(testMap);
	        	  //cvWaitKey(1);
	          }
	          cvNot(DepthMap, DepthMap);
	          

	          cvShowImage("depth", DepthMap);
	         cvWaitKey(200);
	     
	         pp.ReleaseFrame();
	           
			
		}
	}
	public static void templateMatch(IplImage map)
	{
		double[] min_val ,max_val;
		CvPoint maxLoc,minLoc;
		maxLoc=new CvPoint();
		minLoc=new CvPoint();
		min_val= new double[10];
		max_val= new double[10];
		
		
		IplImage map32f= cvCreateImage(cvSize(map.width(), map.height()), IPL_DEPTH_32F, 1);
		IplImage hand= capture; 
		IplImage hand32f= cvCreateImage(cvGetSize(hand), IPL_DEPTH_32F, 1);
		
		IplImage coeff= cvCreateImage(cvSize(map.width()-hand.width()+1, map.height()-hand.height()+1), IPL_DEPTH_32F, 1);
		
		cvCvtScale(map, map32f, 1.0/255.0, 0);
		
		cvConvertImage(hand, hand,CV_BGR2GRAY);
		cvMatchTemplate(map32f, hand32f,coeff, CV_TM_CCOEFF_NORMED);

		cvMinMaxLoc(coeff, min_val, max_val,minLoc,maxLoc,null);
		
		cvDrawRect(DepthMap_3C, maxLoc, cvPoint(maxLoc.x()+hand.width(), maxLoc.y()+hand.height()), cvScalar(255, 0, 0, 0), 2, 0, 0);
		
		//System.out.println(max_val[0]);
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
	           
	         
	        		   
	        		
	        	
	       
	           	  Pointer depth_pt= new ShortPointer(depthmap_R);
		          CvMat depth_mat = cvMat(240, 320, CV_16UC1, depth_pt);
		          cvConvert(depth_mat, DepthMap_R);
		   
		          
		          depth_pt= new ShortPointer(depthmap_G);
		           depth_mat = cvMat(240, 320, CV_16UC1, depth_pt);
		          cvConvert(depth_mat, DepthMap_G);
		        
		          depth_pt= new ShortPointer(depthmap_B);
		          depth_mat = cvMat(240, 320, CV_16UC1, depth_pt);
		          cvConvert(depth_mat, DepthMap_B);
		   
		          depth_pt= new ShortPointer(depthmap_V);
		          depth_mat = cvMat(240, 320, CV_16UC1, depth_pt);
		          cvConvert(depth_mat, DepthMap);
		          cvNot(DepthMap, DepthMap);	
		          
		         
		          cvMerge(DepthMap_B, DepthMap_G,DepthMap_R, null, DepthMap_3C);
		     
		          if(Chkmatch==true){
		        	 // findHand(DepthMap);
		        	  
		        	 // drawSURFpt(DepthMap_3C,objectKeypoints);
		        	  
		        	 // cvExtractSURF(DepthMap, null, imageKeypoints,  imageDescriptors, storage1,params, 0); 
		              
		              
		             // drawSURFpt(DepthMap_3C,imageKeypoints);
		              
		           
		        	// matchingSURFpts(imageKeypoints, imageDescriptors, objectKeypoints, objectDescriptors, DepthMap_3C);
		        	// drawSURFLine(DepthMap_3C, objectKeypoints, imageKeypoints);
		     		
		           //  getCentroid(capture);
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
		          
		          
		          cvShowImage("depth3C", DepthMap_3C);
		          cvShowImage("depth1C", DepthMap);
			      cvShowImage("RGB", RgbMap	);   
			     
			      
		            
		          switch(cvWaitKey(10))
		          {
		          case '0':
		        	  caputureBoxImage(DepthMap,capture);
		        	  FeatureDescriptor fd0 = new FeatureDescriptor();
		        	  CvMat data0=fd0.get1DHistogram(captureArr, 0, 1);
		        	  
		        	  for (int i = 0; i < 576; i++) 
						data.put(0, i, data0.get(0, i));
					
		        	
		        	  break;
		          case '1':
		        	  caputureBoxImage(DepthMap,capture);
		        	  FeatureDescriptor fd1 = new FeatureDescriptor();
		        	  CvMat data1=fd1.get1DHistogram(captureArr, 0, 1);
		        	  
		        	  for (int i = 0; i < 576; i++) 
							data.put(1, i, data1.get(0, i));
						
		        	  break;
		          case '2':
		        	  caputureBoxImage(DepthMap,capture);
		        	  FeatureDescriptor fd2 = new FeatureDescriptor();
		        	  CvMat data2=fd2.get1DHistogram(captureArr, 0, 1);
		        	  
		        	  for (int i = 0; i < 576; i++) 
							data.put(2, i, data2.get(0, i));
						
		        	  break;
		          
		          
		          case 'y':
		        	  Chkmatch=true;
		        	  
		        	  classifier.trainSVM(data,1);
		        	  break;
		        
		          case 'n':
		        	  Chkmatch=false;
		        	  caputureBoxImage(DepthMap,capture);
		        	  FeatureDescriptor fd3 = new FeatureDescriptor();
		        	 
		        	  CvMat data3=fd3.get1DHistogram(captureArr, 0, 1);
		        
		        	 classifier.classify(data3);
		        	  break;
		         
		          case 'f': 
	
		        	  break;
		        	  
		          }
		          
		          
		          pp.ReleaseFrame();
		   }
	}
	public static void caputureBoxImage(IplImage DepthMap,IplImage capture)
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
	
       cvExtractSURF(capture, null, objectKeypoints, objectDescriptors, storage, params, 0);
 
	}
	public static void drawSURFpt(IplImage image,CvSeq keyPt)
	{
		   for (int i = 0; i < keyPt.total(); i++) {
				
		    	CvSURFPoint obj_pt = new CvSURFPoint(cvGetSeqElem(keyPt, i));
					
				CvPoint center= new CvPoint();
				center.put(Math.round(obj_pt.pt().x()), Math.round(obj_pt.pt().y()));
				//int radious = (int) Math.round(obj_pt.size()*1.2/9.0*2.0);
				cvDrawCircle(image, center, 1,cvScalar(0,255, 0, 0),1,0,0);
				
		    }
	}
	public static void findHand(IplImage map)
	{
		
        CvSeq imageKeypoints = new CvSeq();

        CvSeq imageDescriptors = new CvSeq();

        cvExtractSURF(map, null, imageKeypoints,  imageDescriptors, storage1,params, 0); 
        
        drawSURFpt(DepthMap_3C, imageKeypoints);
        
//        matchingSURFpts(imageKeypoints, imageDescriptors, objectKeypoints, objectDescriptors, DepthMap_3C);
        
//		for(int x=0; x<320-CaputureBox_WIDTH; x+=2){
//		for(int y=0; y<240-CaputureBox_HEIGHT; y+=2){
//		
//		ROIBox=cvRect(x, y, CaputureBox_WIDTH, CaputureBox_HEIGHT);
//			
//		cvSetImageROI(map, ROIBox);
//		cvCopy(map, CurrentROI);
//		cvResetImageROI(map);
//		
//        //cvExtractSURF(CurrentROI, null, imageKeypoints,  imageDescriptors, storage,params, 0); 
//         
//        
//       
//       //cvShowImage("ROIBOX", CurrentROI);
//		
//		//cvWaitKey(1);
//		}
//		//fprintln(x);
//		}
	}
	public static double calcVectorDist(FloatBuffer obj_vec, FloatBuffer img_vec, int dim)
	{
		double sum=0;
		for (int i = 0; i < dim; i++) {
			
			sum+=Math.pow(obj_vec.get(i)-img_vec.get(i), 2);
						
		}
		
		
		return  Math.sqrt(sum);
	}
	
	public static void drawSURFLine(IplImage image,CvSeq objectKeypoints,CvSeq imageKeypoints)
	{
		
		
		if((double)map.keySet().size()/(double)objectKeypoints.total()<RATIO){
			return;
		}
		Iterator<Number> iterator =map.keySet().iterator();
		
		
		while(iterator.hasNext()) {
			int key =(Integer) iterator.next();

			CvSURFPoint obj_pt = new CvSURFPoint(cvGetSeqElem(objectKeypoints,key));
			
			CvSURFPoint img_pt = new CvSURFPoint(cvGetSeqElem(imageKeypoints, (Integer)map.get(key)));
			
			CvPoint pt1= cvPoint(Math.round(obj_pt.pt().x()),Math.round(obj_pt.pt().y()));
			CvPoint pt2= cvPoint(Math.round(img_pt.pt().x()),Math.round(img_pt.pt().y()));
			
			cvDrawLine(image, pt1, pt2, cvScalar(0,255, 0, 0), 1, 1, 0);
			
		}
	}
	public static void matchingSURFpts(CvSeq imageKeypoints, CvSeq imageDescriptors, CvSeq objectKeypoints,CvSeq objectDescriptors ,IplImage DepthImage)
	{
		map= new HashMap<Number, Number>();
		CvSURFPoint obj_pt ,img_pt;
		FloatBuffer obj_vec,img_vec;
		int obj_elem_size,img_elem_size;
		
		
		obj_elem_size=objectDescriptors.elem_size();
		img_elem_size=imageDescriptors.elem_size();
		
		for (int i = 0; i < objectDescriptors.total(); i++) {
			int neighbor=-1;
			double minDist=1000000;
			double dist=0;
			obj_pt = new CvSURFPoint(cvGetSeqElem(objectKeypoints, i));
			obj_vec=cvGetSeqElem(objectDescriptors, i).capacity(obj_elem_size).asByteBuffer().asFloatBuffer();
			
			for (int j = 0; j < imageDescriptors.total(); j++) {
		
				img_pt = new CvSURFPoint(cvGetSeqElem(imageKeypoints, j));
				
		      	if(obj_pt.laplacian()==img_pt.laplacian())
		      	{
		      		
		      		img_vec=cvGetSeqElem(imageDescriptors, j).capacity(img_elem_size).asByteBuffer().asFloatBuffer();
		      		
		      		dist=calcVectorDist(obj_vec, img_vec,128);
		      		
		      		if(minDist>dist){
		      			minDist=dist;
		      			neighbor=j;
		      		}
		      	}
				
			}
			
			if(minDist<THRESHOLD && neighbor>-1)
			{
				map.put(i, neighbor);
				//System.out.println(i);
			}
			
			
		}
   
		
       
        
 	}
	
	
	public static void getCentroid(IplImage img)
	{
		CvMoments moments = new CvMoments();
		cvMoments(img,moments,1);
		
		double m00 =cvGetSpatialMoment(moments, 0, 0);
		double m10 =cvGetSpatialMoment(moments, 1, 0);
		double m01 =cvGetSpatialMoment(moments, 0, 1);
		
		if(m00!=0){
		int Xc_hand=(int) Math.round(m10/m00);
		int Yc_hand=(int) Math.round(m01/m00);
		
	
		
		CvFont font = new CvFont();
		
		cvInitFont(font, CV_FONT_HERSHEY_COMPLEX, 0.5, 0.5,0,0,0	);
		cvPutText(DepthMap_3C, "1CENTER1" , cvPoint(Xc_hand, Yc_hand), font, CV_RGB(0,0,255));	
		

		cvDrawCircle(DepthMap_3C, cvPoint(Xc_hand, Yc_hand), 2, cvScalar(255, 255, 255, 0), 3, CV_AA, 0);
		
		cvDrawCircle(DepthMap_3C, cvPoint(Xc_hand, Yc_hand), 36, cvScalar(255, 255, 255, 0), 1, CV_AA, 0);
		
		
		
		
		
		}
		
	}


	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		
        	init();
        
	      // makeTestSet();
	       loadTestSet();
        	//realTimeShow();
        	pp.Close();

	        System.exit(0);

	}

}
