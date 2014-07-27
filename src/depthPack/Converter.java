package depthPack;


import static com.googlecode.javacv.cpp.opencv_core.*;

import com.googlecode.javacpp.*;

public class Converter {

	private  short[] depthmap;
	private static CvMat depth_mat;
	public Converter()
	{
		//nothing happen
		 
	}
	public IplImage CvtArr2Img(IplImage DepthMap,short[] depthArr,int width, int height)
	{     
		int x;
		int y;
		for (int i = 0; i < depthArr.length; i++) {
			
			x=i%width;
		    y=i/width;
			
		cvSetReal2D(DepthMap, y, x, depthArr[i]);
		}
		
        return DepthMap;
	}
	public short[] CvtImg2Arr(IplImage img)
	{
	
		depthmap = new short[img.height()*img.width()];
		
		for (int i = 0; i < img.width(); i++) {
			for (int j = 0; j < img.height(); j++) {
			
				depthmap[i+j*img.width()]=(short) cvGetReal2D(img, j, i);
				
			}	
		}
		
		return depthmap;
	}

	
	
}
