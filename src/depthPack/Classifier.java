package depthPack;

import com.googlecode.javacv.cpp.opencv_core.CvTermCriteria;
import com.googlecode.javacv.cpp.opencv_ml.CvSVM;
import com.googlecode.javacv.cpp.opencv_ml.CvSVMParams;
import static com.googlecode.javacv.cpp.opencv_core.*;

public class Classifier {
	
	CvSVM svm;
	CvSVMParams param;
	
	public Classifier()
	{
		svm = new CvSVM();
	
		CvTermCriteria criteria = cvTermCriteria (CV_TERMCRIT_EPS, 100, 1e-6);
		
		param 
		= new CvSVMParams(CvSVM.C_SVC, CvSVM.LINEAR, 10, 8, 1, 10, 0.5, 0.1, null, criteria	);
		
	}
	
	
	private void train()
	{
		 
	}
}
