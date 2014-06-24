package depthPack;

import static com.googlecode.javacv.cpp.opencv_core.IPL_DEPTH_8U;
import static com.googlecode.javacv.cpp.opencv_core.cvCopy;
import static com.googlecode.javacv.cpp.opencv_core.cvCreateImage;
import static com.googlecode.javacv.cpp.opencv_core.cvRect;
import static com.googlecode.javacv.cpp.opencv_core.cvResetImageROI;
import static com.googlecode.javacv.cpp.opencv_core.cvSetImageROI;
import static com.googlecode.javacv.cpp.opencv_core.cvSize;
import static com.googlecode.javacv.cpp.opencv_imgproc.CV_GAUSSIAN;
import static com.googlecode.javacv.cpp.opencv_imgproc.cvSmooth;

import com.googlecode.javacv.cpp.opencv_core.*;

public class HandFinder {
	private static CvRect HandBox;
	private static IplImage HandArea;
	private static Converter cvt;
	private static short[] HandArr;
	public HandFinder()
	{
		 cvt = new Converter();
		 HandArea = cvCreateImage(cvSize(90, 90), IPL_DEPTH_8U, 1);
		 HandArr  = new short[90*90];
	}
	
	public short[] FindHandSW(IplImage img,int x ,int y,short[] depthmap_V)
	{
		
				HandBox = cvRect(x, y, 90, 90);
				cvSetImageROI(img, HandBox);
				cvCopy(img, HandArea);
				cvResetImageROI(img);
				
				
				cvSmooth(HandArea, HandArea, CV_GAUSSIAN, 3);
				
				
				for(int i=0,j=0; i< main.MATSIZE;i++)
				{
					if((i%320)>=x &&(i%320)<main.CaputureBox_WIDTH+x && (i/320)>=y &&(i/320)<main.CaputureBox_HEIGHT+y)
					//if((i%320)<CaputureBox_WIDTH && (i/320)<CaputureBox_HEIGHT)
					{
						HandArr[j++]=depthmap_V[i];
					}
				}
		//return	cvt.CvtImg2Arr(HandArea);
		return HandArr;
		
		
	}
	public IplImage getImg()
	{
		return HandArea;
	}
}
