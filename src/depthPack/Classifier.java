package depthPack;


import static com.googlecode.javacv.cpp.opencv_highgui.*;
import com.googlecode.javacv.cpp.opencv_core.CvTermCriteria;
import com.googlecode.javacv.cpp.opencv_ml.CvBoost;
import com.googlecode.javacv.cpp.opencv_ml.CvBoostParams;
import com.googlecode.javacv.cpp.opencv_ml.CvSVM;
import com.googlecode.javacv.cpp.opencv_ml.CvSVMParams;
import static com.googlecode.javacv.cpp.opencv_core.*;

public class Classifier {
	private static final int DATSIZE = 576;
	
	private static CvSVM svm;

	private static CvSVMParams param_svm;
	private static CvMat temp,temp_cls;
	public Classifier()
	{
		
	 
	  initSVM();
		
	}
	public void initSVM()
	{
		svm = new CvSVM();
		CvTermCriteria criteria = cvTermCriteria (CV_TERMCRIT_EPS, 1000, 1e-6);
		
		param_svm 
		= new CvSVMParams();
		param_svm.svm_type(CvSVM.C_SVC);
		param_svm.kernel_type(CvSVM.LINEAR);
		param_svm.degree(9);
		param_svm.gamma(8);
		param_svm.coef0(0);
		param_svm.C(100);
		param_svm.nu(0.9);
		param_svm.p(0);
		param_svm.class_weights(null);
		param_svm.term_crit(criteria);
		
		temp= cvCreateMat(2, DATSIZE, CV_32FC1);
		temp_cls=cvCreateMat(2, 1, CV_32SC1);
	}

	public static CvSVM getSVM()
	{
		return svm;
	}
	
	
	public void trainSVM(CvMat Data,int EntrySize,int classnum)
	{
		
		
		CvMat temp_cls=cvCreateMat(EntrySize, 1, CV_32SC1);
		
		for (int j = 0; j < EntrySize; j++) {
			if(j<EntrySize/2)
				temp_cls.put(j, 0,classnum);
			else
				temp_cls.put(j, 0,-1);
				
		}
		
		 svm.train(Data, temp_cls, null, null, param_svm);
		
		 cvReleaseMat(temp);
		 cvReleaseMat(temp_cls);
		 
	}
	public float classifySVM(CvMat Data)
	{
		CvMat temp= cvCreateMat(1, DATSIZE, CV_32FC1);
		
		 for (int i = 0; i < DATSIZE; i++) 
			 temp.put(0,i,(float)Data.get(0, i));
		 
		 
		float response=svm.predict(temp,false);
		
		return response;
	}
	
	
}
