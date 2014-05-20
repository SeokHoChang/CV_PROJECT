package depthPack;


import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvSmooth;

import com.googlecode.javacpp.*;
import com.googlecode.javacv.*;
import com.googlecode.javacv.cpp.*;
public class Converter {
	private static IplImage DepthMap ;

	
	public Converter()
	{

     	
	
 
	}
	public IplImage CvtArr2Img(IplImage DepthMap,short[] depthArr,int width, int height)
	{
//		DepthMap = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		Pointer depth_pt= new ShortPointer(depthArr);
        CvMat depth_mat = cvMat(height, width, CV_16UC1,depth_pt);
        cvConvert(depth_mat, DepthMap);
        
        return DepthMap;
	}
	public short[] CvtImg2Arr(IplImage img)
	{
	
		short[] depthmap = new short[img.height()*img.width()];
		
		for (int i = 0; i < img.width(); i++) {
			for (int j = 0; j < img.height(); j++) {
			
				depthmap[i+j*img.width()]=(short) cvGetReal2D(img, j, i);
				
			}	
		}
		
		return depthmap;
	}

	
	
}
