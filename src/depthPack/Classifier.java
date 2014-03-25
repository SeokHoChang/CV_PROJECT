package depthPack;



import com.googlecode.javacv.cpp.opencv_core.CvTermCriteria;
import com.googlecode.javacv.cpp.opencv_ml.CvSVM;
import com.googlecode.javacv.cpp.opencv_ml.CvSVMParams;
import static com.googlecode.javacv.cpp.opencv_core.*;

public class Classifier {
	
	private static CvSVM svm;
	private static CvSVMParams param;
	
	public Classifier()
	{
		svm = new CvSVM();
	
		CvTermCriteria criteria = cvTermCriteria (CV_TERMCRIT_EPS, 1000, 1e-6);
		
		param 
		= new CvSVMParams();
		param.svm_type(CvSVM.C_SVC);
		param.kernel_type(CvSVM.RBF);
		param.degree(100);
		param.gamma(20);
		param.coef0(0);
		param.C(7);
		param.nu(0);
		param.p(0);
		param.class_weights(null);
		param.term_crit(criteria);
	}
	
	
	public void trainSVM(CvMat Data,int i)
	{
		
		CvMat temp= cvCreateMat(3, 576, CV_32FC1);
		temp.put(Data);
		
		CvMat temp_cls=cvCreateMat(3, 1, CV_32SC1);
		temp_cls.put(0, 0,0);
		temp_cls.put(1, 0,1);
		temp_cls.put(2, 0,2);
			
		
		 svm.train(Data, temp_cls, null, null, param);
		 
		 
	}
	public void classify(CvMat Data)
	{

		System.out.println(svm.predict(Data,false));
	}
}
