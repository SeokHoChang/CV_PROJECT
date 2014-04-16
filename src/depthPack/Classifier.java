package depthPack;


import static com.googlecode.javacv.cpp.opencv_highgui.*;
import com.googlecode.javacv.cpp.opencv_core.CvTermCriteria;
import com.googlecode.javacv.cpp.opencv_ml.CvBoost;
import com.googlecode.javacv.cpp.opencv_ml.CvBoostParams;
import com.googlecode.javacv.cpp.opencv_ml.CvSVM;
import com.googlecode.javacv.cpp.opencv_ml.CvSVMParams;
import static com.googlecode.javacv.cpp.opencv_core.*;

public class Classifier {
	
	private static CvSVM svm;
	private static CvBoost boost;
	private static CvBoostParams param_boost;
	private static CvSVMParams param_svm;
	
	public Classifier()
	{
		
	  initBOOST();
	  initSVM();
		
	}
	public void initSVM()
	{
		svm = new CvSVM();
		CvTermCriteria criteria = cvTermCriteria (CV_TERMCRIT_EPS, 1000, 1e-6);
		
		param_svm 
		= new CvSVMParams();
		param_svm.svm_type(CvSVM.C_SVC);
		param_svm.kernel_type(CvSVM.RBF);
		param_svm.degree(9);
		param_svm.gamma(8);
		param_svm.coef0(0);
		param_svm.C(10);
		param_svm.nu(0.8);
		param_svm.p(0);
		param_svm.class_weights(null);
		param_svm.term_crit(criteria);
	}
	public void initBOOST()
	{
		boost = new CvBoost();
		param_boost = new CvBoostParams();
		param_boost.boost_type(CvBoost.REAL);
		param_boost.weak_count(10);
		param_boost.weight_trim_rate(0.95);
		param_boost.max_depth(3);
		param_boost.use_surrogates(false);
		param_boost.priors(null);
		param_boost.split_criteria(CvBoost.DEFAULT);
	}
	
	public void trainBOOST(CvMat Data,int i)
	{
		CvMat temp= cvCreateMat(2, 576, CV_32FC1);
		temp.put(Data);
		
		CvMat temp_cls=cvCreateMat(2, 1, CV_32SC1);
		temp_cls.put(0, 0,0);
		temp_cls.put(1, 0,1);
		//temp_cls.put(2, 0,2);
		//temp_cls.put(2, 0,3);	
		
		boost.train(temp,1, temp_cls, null, null, null, null, param_boost, false);
		 
		 cvReleaseMat(temp);
		 cvReleaseMat(temp_cls);
	}
	public void classifyBOOST(CvMat Data)
	{
		CvMat temp= cvCreateMat(2, 576, CV_32FC1);
		temp.put(Data);
		System.out.println(boost.predict(temp, null,null, null, false, false));
	}
	public void trainSVM(CvMat Data,int i)
	{
		
		CvMat temp= cvCreateMat(2, 576, CV_32FC1);
		
		temp.put(Data);
		
		CvMat temp_cls=cvCreateMat(2, 1, CV_32SC1);
		temp_cls.put(0, 0,0);
		temp_cls.put(1, 0,1);
		//temp_cls.put(2, 0,2);
		//temp_cls.put(2, 0,3);	
		
		 svm.train(temp, temp_cls, null, null, param_svm);
		 
		 cvReleaseMat(temp);
		 cvReleaseMat(temp_cls);
		 
	}
	public void classifySVM(CvMat Data)
	{
		CvMat temp= cvCreateMat(1, 576, CV_32FC1);
		 for (int i = 0; i < 576; i++) 
			 temp.put(0,i,(float)Data.get(0, i)/(float)100);
		 
		System.out.println(svm.predict(temp,true));
	}
	
	public void testCode()
	{
		IplImage img = cvCreateImage(cvSize(1000, 1000), IPL_DEPTH_8U, 3);
		cvZero(img);
		
		int sample_count = 1000;
		
		CvMat train_data= cvCreateMat(sample_count, 2, CV_32FC1);
		CvMat train_class= cvCreateMat(sample_count, 1, CV_32SC1);
		
		for (int i = 0; i < sample_count; i++) {
			
			float x = (int) (Math.random()*1000);
			float y = (int) (Math.random()*1000);
			
			train_data.put(i,0,x/1000);
			train_data.put(i,1,y/1000);
			int c = (y > 200*Math.cos(x*3.14/300) + 400) ? ((x > 600) ? 0 : 1) : ((x > 400) ? 2 : 3);
			train_class.put(i, 0, c);
			
		
		}
		
		svm.train(train_data, train_class, null,null, param_svm);
		
		for (int x=0; x<img.width(); x++) {
			for (int y=0; y<img.height(); y++) {
				CvMat sample = cvCreateMat(1, 2, CV_32FC1);
				sample.put(0, 0,(float)x/1000);
				sample.put(0, 1,(float)y/1000);
				
				float response = svm.predict(sample, true);
				
				if(response==0)
					cvSet2D(img, y, x, cvScalar(180, 0, 0, 0));
				else
					cvSet2D(img, y, x, cvScalar(0, 255, 0, 0));
					
				
				
			}
		}
		
		for (int i = 0; i < sample_count; i++) {
		int x = (int) Math.round(train_data.get(i, 0)*1000);
		int y = (int) Math.round(train_data.get(i, 1)*1000);
		int c = (int) Math.round(train_class.get(i, 0));
		
		if(c==0)
			cvCircle(img, cvPoint(x, y), 2, cvScalar(0, 0, 0, 0), 0, 1, 1);
		else
			cvCircle(img, cvPoint(x, y), 2, cvScalar(255, 255,255, 0), 0, 1, 1);
		
		}
		
		cvShowImage("result", img);
	
		cvWaitKey(10000	);
		
	}
}
