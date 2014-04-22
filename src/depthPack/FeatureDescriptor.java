package depthPack;

import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_highgui.*;
import java.awt.Point;


import com.googlecode.javacv.cpp.opencv_core.IplImage;



public class FeatureDescriptor {
	
	private static final int WEIGHTSCALE=1;
	private static final int RADIUS = 36;
	private static final int FI = 360;
	private static final int ANGLE_SIZE=45;
	private static final int ANGLE_BIN_SIZE = 8;
	private static final int RADIUS_BIN_SIZE = 4;
	private static final int GRAD_BIN_SIZE_XY = 8;
	private static final int GRAD_BIN_SIZE_YZ = 5;
	private static final int GRAD_BIN_SIZE_XZ = 5;
	
	
	private static final int TRAINING_MAT_SIZE= ANGLE_BIN_SIZE*RADIUS_BIN_SIZE*(GRAD_BIN_SIZE_XY+GRAD_BIN_SIZE_XZ+GRAD_BIN_SIZE_YZ);
	
	private static final int WIDTH=90;
	private static final int HEIGHT=90;
	
	private static final int MAT_ARRAY_SIZE=WIDTH*HEIGHT;
	
	private static final int PROJECTED_NORM_XY = 0;
	private static final int PROJECTED_NORM_YZ = 1;
	private static final int PROJECTED_NORM_XZ = 2;

	private static final int MANHATTAN_DISTANCE_MODE = 0;

	private static CvMat data;
	private static CvMat[] NormMatArr;
	private static double[][][] HistogramXY;
	private static double[][][] HistogramYZ;
	private static double[][][] HistogramXZ;
	
	
	public FeatureDescriptor()
	{
		data= cvCreateMat(1,TRAINING_MAT_SIZE , CV_32FC1);
		cvSetZero(data);
		for (int i = 0; i < TRAINING_MAT_SIZE; i++) {
			data.put(0, i, 0);
		}
		
		NormMatArr= new CvMat[MAT_ARRAY_SIZE];
		for (int i = 0; i < NormMatArr.length; i++) {
			NormMatArr[i]= cvCreateMat(3, 1, CV_32FC1);
			cvSetZero(NormMatArr[i]);
			
		}
		
		HistogramXY= new double[ANGLE_BIN_SIZE][RADIUS_BIN_SIZE][GRAD_BIN_SIZE_XY];
		HistogramYZ= new double[ANGLE_BIN_SIZE][RADIUS_BIN_SIZE][GRAD_BIN_SIZE_YZ];
		HistogramXZ= new double[ANGLE_BIN_SIZE][RADIUS_BIN_SIZE][GRAD_BIN_SIZE_XZ];
		
		for(int i=0;i<ANGLE_BIN_SIZE;i++)
			for (int j = 0; j < RADIUS_BIN_SIZE; j++) {
				for (int j2 = 0; j2 < GRAD_BIN_SIZE_XY; j2++) {
					HistogramXY[i][j][j2]=0;
				}
				for (int j2 = 0; j2 < GRAD_BIN_SIZE_YZ; j2++) {
					HistogramYZ[i][j][j2]=0;
				}
				for (int j2 = 0; j2 < GRAD_BIN_SIZE_XZ; j2++) {
					HistogramXZ[i][j][j2]=0;
				}
			}
		
		
	}
	
	public CvMat get1DHistogram(short[] DepthMap,int p , int l)
	{
//		for (int i = 0; i < DepthMap.length; i++) {
//			System.out.println(DepthMap[i]);
//		}
		
		makeNormMatArr(DepthMap);
		fillHIST(NormMatArr);
		makeHistTo1DMat();
		
		if(l==1)
		showHIST();
		return data;
	}
	private void showHIST()
	{
		IplImage hist = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);
		cvSetZero(hist);
		
		
		for (int i = 0; i < TRAINING_MAT_SIZE; i++) {
			//System.out.println(data.get(0, i));
			
			
			if(i<ANGLE_BIN_SIZE*RADIUS_BIN_SIZE*GRAD_BIN_SIZE_XY)
			{ 
				cvLine(hist,cvPoint(10+i, 0), cvPoint(10+i,(int)Math.round(data.get(0, i))), CV_RGB(0,255,0), 1, CV_AA, 0);
				
			continue;
			}
			if(i>=ANGLE_BIN_SIZE*RADIUS_BIN_SIZE*GRAD_BIN_SIZE_XY && i<TRAINING_MAT_SIZE-ANGLE_BIN_SIZE*RADIUS_BIN_SIZE*GRAD_BIN_SIZE_XZ)
				{
					cvLine(hist,cvPoint(20+i, 0), cvPoint(20+i,(int)Math.round(data.get(0, i))), CV_RGB(255, 0,0), 1, CV_AA, 0);
					
					continue;
				}
			
			if( i>=TRAINING_MAT_SIZE-ANGLE_BIN_SIZE*RADIUS_BIN_SIZE*GRAD_BIN_SIZE_XZ)
				{
				
					cvLine(hist,cvPoint(30+i, 0), cvPoint(30+i,(int)Math.round(data.get(0, i))), CV_RGB(0, 0,255), 1, CV_AA, 0);
					
					continue;
				}
			
		}
		cvShowImage("hist",hist);
		cvReleaseImage(hist);
		
	}
	
	private void makeHistTo1DMat()
	{
		//for (int i = 0; i < TRAINING_MAT_SIZE; i++) 
			//System.out.println(data.get(0, i));
			
		int idx=0;
	for (int cnt = 0; cnt < 3; cnt++) 
		for(int i=0;i<ANGLE_BIN_SIZE;i++)
			for (int j = 0; j < RADIUS_BIN_SIZE; j++) {
				
				
				switch (cnt) {
				case 0:
					for (int k = 0; k < GRAD_BIN_SIZE_XY; k++) 
						data.put(0, idx++, HistogramXY[i][j][k]);
				
					break;
				case 1:
					for (int k = 0; k < GRAD_BIN_SIZE_YZ; k++) 
						data.put(0, idx++, HistogramYZ[i][j][k]);
				
					break;
				case 2:
					for (int k = 0; k < GRAD_BIN_SIZE_XZ; k++) 
						data.put(0, idx++, HistogramXZ[i][j][k]);
				
					break;
				default:
					break;
				}
				
				
			}
		
	}
	
	
	private void fillHIST(CvMat[] NormMatArr)
	{
		double[] BinNumXY,BinNumXZ,BinNumYZ;
	
		BinNumXY= new double[3];
		BinNumYZ= new double[3];
		BinNumXZ= new double[3];
		
		for (int j = 1; j < HEIGHT-1; j++) 
			for (int k = 1; k < WIDTH-1; k++)
			 {
				int i = k+j*WIDTH;
				
			
			CvMat ProjectedNormXY = getProjectVector(NormMatArr[i], PROJECTED_NORM_XY);
			CvMat ProjectedNormYZ = getProjectVector(NormMatArr[i], PROJECTED_NORM_YZ);
			CvMat ProjectedNormXZ = getProjectVector(NormMatArr[i], PROJECTED_NORM_XZ);
			
			
			BinNumXY= getBinNum(ProjectedNormXY, PROJECTED_NORM_XY);
			BinNumYZ= getBinNum(ProjectedNormYZ, PROJECTED_NORM_YZ);
			BinNumXZ= getBinNum(ProjectedNormXZ, PROJECTED_NORM_XZ);
			
		
			double[] weightsXY= calcWeight(BinNumXY[0], BinNumXY[1], BinNumXY[2]);
			double[] weightsYZ= calcWeight(BinNumYZ[0], BinNumYZ[1], BinNumYZ[2]);
			double[] weightsXZ= calcWeight(BinNumXZ[0], BinNumXZ[1], BinNumXZ[2]);
			 
			int idx_ANGLE,idx_RADIUS;
			if((idx_ANGLE= getAngleBin(k,j))==-1){
				//System.out.println("(x,y)= ("+k+","+j+")");
				continue;
			}
			
			if((idx_RADIUS= getRadiusBin(k,j))==-1)
			{
				//System.out.println("(x,y)= ("+k+","+j+")");
				continue;
			}
			
			if(!(ProjectedNormXY.get(0, 0)==0 &&ProjectedNormXY.get(0, 1)==0)){
			HistogramXY[idx_ANGLE][idx_RADIUS][(int) BinNumXY[0]]+=weightsXY[0]*WEIGHTSCALE;
			HistogramXY[idx_ANGLE][idx_RADIUS][(int) BinNumXY[1]]+=weightsXY[1]*WEIGHTSCALE;
			}
			if(!(ProjectedNormYZ.get(0, 0)==0 &&ProjectedNormYZ.get(0, 1)==0)){
			HistogramYZ[idx_ANGLE][idx_RADIUS][(int) BinNumYZ[0]]+=weightsYZ[0]*WEIGHTSCALE;
			HistogramYZ[idx_ANGLE][idx_RADIUS][(int) BinNumYZ[1]]+=weightsYZ[1]*WEIGHTSCALE;
			}
			if(!(ProjectedNormXZ.get(0, 0)==0 &&ProjectedNormXZ.get(0, 1)==0)){
			HistogramXZ[idx_ANGLE][idx_RADIUS][(int) BinNumXZ[0]]+=weightsXZ[0]*WEIGHTSCALE;
			HistogramXZ[idx_ANGLE][idx_RADIUS][(int) BinNumXZ[1]]+=weightsXZ[1]*WEIGHTSCALE;
			}
			
		}
				
		
	}
	
	private int getRadiusBin(int x, int y)
	{
		
		
		double distance=Math.sqrt(Math.pow(x-WIDTH/2,2)+Math.pow(y-HEIGHT/2,2));
		
		for (int i = 0; i < RADIUS_BIN_SIZE; i++) {
			
			if(RADIUS/RADIUS_BIN_SIZE*i<=distance && distance<RADIUS/RADIUS_BIN_SIZE *(i+1))
				return i;
		}	
		
		return -1;
	}
	
	private int getAngleBin(int x, int y)
	{
		
		
		double Angle=0;
		
			x=x-WIDTH/2;
			y=y-HEIGHT/2;
		
		  if(( Angle=(Math.atan2(y, x)/Math.PI) *180)<0)
			 Angle=360+Angle;
		  
		  
		
	for (int i = 0; i < ANGLE_BIN_SIZE; i++) {
			
			if(FI/ANGLE_BIN_SIZE*i<=Angle && Angle<FI/ANGLE_BIN_SIZE *(i+1))
				return i;
		}	
			return -1;
				
	}

	private double[] calcWeight(double num_bin0, double num_bin1, double angle)
	{
		double[] theta= new double[2];
		
		
		theta[0]=angle-ANGLE_SIZE*num_bin0;
		theta[1]=ANGLE_SIZE*num_bin1-angle;
		
//		System.out.println("ANGLE: "+angle);
//		System.out.println("BIN: "+num_bin0+" , "+num_bin1);
//		System.out.println("THETA: "+theta[0]+" , "+theta[1]);
//		System.out.println("SINE: "+Math.sin((theta[0]/(double)180)*Math.PI)+" , "+Math.sin((theta[1]/(double)180)*Math.PI));
//		
		double[] weights= new double[2];
		
		
		if((weights[0]=Math.sin((theta[1]/(double)180)*Math.PI)/(Math.sin((theta[0]/(double)180)*Math.PI)+Math.sin((theta[1]/(double)180)*Math.PI)))<0)
			{
				System.out.println(weights[0]);
			 	System.err.println("weight0 cannot be a negative value");
			 	System.exit(1);
			}
		if((weights[1]=Math.sin((theta[0]/(double)180)*Math.PI)/(Math.sin((theta[0]/(double)180)*Math.PI)+Math.sin((theta[1]/(double)180)*Math.PI)))<0)
			{
				System.out.println(weights[1]);
				System.err.println("weight1 cannot be a negative value");
				System.exit(1);
			}
		
//			System.out.println("binNUM:"+num_bin0);
//			System.out.println("weight:"+weights[0]);
//			
//			System.out.println("binNUM:"+num_bin1);
//			System.out.println("weight:"+weights[1]);
//			
			
		return weights;
		
	}
	
	private double[] getBinNum(CvMat vector, int mode)
	{
		int binSize=0;
		double[] BinNum= new double[3];
		double x=0,y=0,theta=0;
		
		switch(mode)
		{
		case PROJECTED_NORM_XY:
			
			x=vector.get(0, 0);
			y=vector.get(1, 0);
			binSize=GRAD_BIN_SIZE_XY;
			break;
		case PROJECTED_NORM_YZ:
		
			x=vector.get(1, 0);
			y=vector.get(2, 0);
			binSize=GRAD_BIN_SIZE_YZ;
			break;
		case PROJECTED_NORM_XZ:
			
			x=vector.get(0, 0);
			y=vector.get(2, 0);
			binSize=GRAD_BIN_SIZE_XZ;
			break;
		default :	
			System.err.println("getBinNum ERROR");
								
		}
		
		
		
		///get vector's angle
		if((theta=(Math.atan2(y, x)/Math.PI) *180)<0)
			theta=360+theta;
	
		if(theta==360) 
			theta=0;
			
		
		if((binSize==5) && (theta>180))
		{
			System.err.println("not possible angle in binSize=5 :"+theta+", "+x+", "+y);
			System.exit(1);
			
		}
			
		
		
		///put bin number 
		for (int i = 0; i < binSize; i++) {
			
			if(ANGLE_SIZE *i<theta && theta<=ANGLE_SIZE*(i+1))
			{
				BinNum[0]=i; 
				
				if(i!=GRAD_BIN_SIZE_XY-1)
					BinNum[1]=i+1;
				else
					BinNum[1]=0;
				
				break;
			}
			
		}
		
		///put angle
		BinNum[2]=theta;
		
		
		
		
		return BinNum;
	}
	private void makeNormMatArr(short[] depthMap)
	{
		int imageHeight = HEIGHT;
		int imageWidth  = WIDTH;
		
		
		for (int i = 0; i < imageWidth; i++) {
			for (int j = 0; j < imageHeight; j++) {
				
				if(i==0 || j==0 || i==WIDTH-1 || j== HEIGHT-1)
				{
					CvMat mat=cvCreateMat(3, 1, CV_32FC1);
					cvSetZero(mat);
					NormMatArr[i+j*WIDTH]=mat;
					continue;
					
				}
				
				
				CvMat pt0 = cvCreateMat(3,1,CV_32FC1);
				CvMat pt1 = cvCreateMat(3,1,CV_32FC1);
				CvMat pt2 = cvCreateMat(3,1,CV_32FC1);
				CvMat pt3 = cvCreateMat(3,1,CV_32FC1);
				
//		
				//dx
				pt0.put(0, 0, i);//up
				pt1.put(0, 0, i);//down
				pt2.put(0, 0, i-1);//left
				pt3.put(0, 0, i+1);//right
				
				//dy
				
				pt0.put(1, 0, j-1);//up
				pt1.put(1, 0, j+1);//down
				pt2.put(1, 0, j);//left
				pt3.put(1, 0, j);//right
				
				//dz
				pt0.put(2, 0, depthMap[i+(j-1)*WIDTH]);//up
				pt1.put(2, 0, depthMap[i+(j+1)*WIDTH]);//down
				pt2.put(2, 0, depthMap[(i-1)+j*WIDTH]);//left
				pt3.put(2, 0, depthMap[(i+1)+j*WIDTH]);//right
				
				//System.out.println(i+j*WIDTH);
				CvMat PlaneNorm = getFittedPlaneNorm(pt0, pt1, pt2, pt3);
				
		
				NormMatArr[i+j*WIDTH]=PlaneNorm;
				
//			
			}
			 
		}
	}
	
	private CvMat getFittedPlaneNorm(CvMat pt0,CvMat pt1, CvMat pt2,CvMat pt3)
	{
		
		CvMat v1 = cvCreateMat(3,1,CV_32FC1);
		CvMat v2 = cvCreateMat(3,1,CV_32FC1);
		CvMat normal= cvCreateMat(3, 1, CV_32FC1);
		
		
		for (int i = 0; i <3; i++) {
		
			
				v1.put(i, 0, pt1.get(i, 0)-pt0.get(i, 0));
			
				v2.put(i, 0,  pt3.get(i, 0)-pt2.get(i, 0));
			
		}
		
		
		cvCrossProduct(v1, v2, normal);
//		System.out.println("v1 : ("+v1.get(0, 0)+", " +v1.get(0, 1)+", "+v1.get(0, 2)+")");
//		System.out.println("v2 : ("+v2.get(0, 0)+", " +v2.get(0, 1)+", "+v2.get(0, 2)+")");
//		
//		
//		System.out.println("normal : ("+normal.get(0, 0)+", " +normal.get(0, 1)+", "+normal.get(0, 2)+")");
//		
		
		if(normal.get(2, 0)<0){
			
			normal.put(0, 0, normal.get(0, 0)*(-1));
			normal.put(1, 0, normal.get(1, 0)*(-1));
			normal.put(2, 0, normal.get(2, 0)*(-1));
	
		}
		
		cvReleaseMat(v1);
		cvReleaseMat(v2);
		
		return normal;
	}
	private CvMat getProjectVector(CvMat norm, int mode)
	{
		switch(mode)
		{
		case PROJECTED_NORM_XY:
			CvMat vectorXY =cvCreateMat(3,1,CV_32FC1);
			
			vectorXY.put(0, 0, norm.get(0,0));
			vectorXY.put(1, 0, norm.get(1,0));
			vectorXY.put(2, 0, 0);
			
			return vectorXY;
			
		case PROJECTED_NORM_YZ:
			CvMat vectorYZ =cvCreateMat(3,1,CV_32FC1);
			
			vectorYZ.put(0, 0, 0);
			vectorYZ.put(1, 0, norm.get(1,0));
			vectorYZ.put(2, 0, norm.get(2,0));
			
			return vectorYZ;
			
			
		case PROJECTED_NORM_XZ:
			CvMat vectorXZ =cvCreateMat(3,1,CV_32FC1);
			
			vectorXZ.put(0, 0, norm.get(0,0));
			vectorXZ.put(1, 0, 0);
			vectorXZ.put(2, 0, norm.get(2,0));
			
			return vectorXZ;
			
		default :
			
			return null;
			
					
		}
		
		
	}
	
}
