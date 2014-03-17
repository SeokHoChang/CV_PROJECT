package depthPack;

import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_highgui.*;
import java.awt.Point;


import com.googlecode.javacv.cpp.opencv_core.IplImage;



public class FeatureDescriptor {
	

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
		
		NormMatArr= new CvMat[MAT_ARRAY_SIZE];
		for (int i = 0; i < NormMatArr.length; i++) {
			NormMatArr[i]= cvCreateMat(3, 1, CV_32FC1);
		}
		
		HistogramXY= new double[ANGLE_BIN_SIZE][RADIUS_BIN_SIZE][GRAD_BIN_SIZE_XY];
		HistogramYZ= new double[ANGLE_BIN_SIZE][RADIUS_BIN_SIZE][GRAD_BIN_SIZE_YZ];
		HistogramXZ= new double[ANGLE_BIN_SIZE][RADIUS_BIN_SIZE][GRAD_BIN_SIZE_XZ];
		
	}
	
	public CvMat get1DHistogram(short[] DepthMap,int p , int l)
	{
		
		makeNormMatArr(DepthMap);
		fillHIST(NormMatArr);
		makeHistTo1DMat();
		showHIST();
		return data;
	}
	private void showHIST()
	{
		IplImage hist = cvCreateImage(cvSize(320, 240), IPL_DEPTH_8U, 3);
		
		for (int i = 0; i < TRAINING_MAT_SIZE; i++) {
			
			if(i<ANGLE_BIN_SIZE*RADIUS_BIN_SIZE*GRAD_BIN_SIZE_XY)
			{ 
				cvLine(hist,cvPoint(10+i, 220), cvPoint(10+i,(int)Math.round(220-data.get(0, i))), CV_RGB(255, 0,0), 1, CV_AA, 0);
				//System.out.println((int)Math.round(220-data.get(0, i)));
			}
			else if(i>ANGLE_BIN_SIZE*RADIUS_BIN_SIZE*GRAD_BIN_SIZE_XY && i<TRAINING_MAT_SIZE-ANGLE_BIN_SIZE*RADIUS_BIN_SIZE*GRAD_BIN_SIZE_XZ)
				{
					cvLine(hist,cvPoint(10+i, 220), cvPoint(10+i,(int)Math.round(220-data.get(0, i))), CV_RGB(0, 255,0), 1, CV_AA, 0);
					//System.out.println(data.get(0, i));
				}
			
			else
				{
				
					cvLine(hist,cvPoint(10+i, 220), cvPoint(10+i,(int)Math.round(220-data.get(0, i))), CV_RGB(0, 0,255), 1, CV_AA, 0);
					//System.out.println((int)Math.round(220-data.get(0, i)));
				}
			
		}
		cvShowImage("hist",hist);
		
	}
	
	private void makeHistTo1DMat()
	{
		data= cvCreateMat(1,TRAINING_MAT_SIZE , CV_32FC1);
		
	for (int cnt = 0; cnt < 3; cnt++) 
		for(int i=0;i<ANGLE_BIN_SIZE;i++)
			for (int j = 0; j < RADIUS_BIN_SIZE; j++) {
				
				
				switch (cnt) {
				case 0:
					for (int k = 0; k < GRAD_BIN_SIZE_XY; k++) {
						int idx= i*(RADIUS_BIN_SIZE*GRAD_BIN_SIZE_XY)+j*GRAD_BIN_SIZE_XY+k;
						data.put(0, idx, HistogramXY[i][j][k]);
						
					}
					
					break;
				case 1:
					for (int k = 0; k < GRAD_BIN_SIZE_YZ; k++) {
						int idx= ANGLE_BIN_SIZE*RADIUS_BIN_SIZE*GRAD_BIN_SIZE_XY
								+i*(RADIUS_BIN_SIZE*GRAD_BIN_SIZE_YZ)+j*GRAD_BIN_SIZE_YZ+k;
						
						data.put(0, idx, HistogramYZ[i][j][k]);
						
						
					}
					break;
				case 2:
					for (int k = 0; k < GRAD_BIN_SIZE_XZ; k++) {
						int idx= ANGLE_BIN_SIZE*RADIUS_BIN_SIZE*GRAD_BIN_SIZE_XY
								+ANGLE_BIN_SIZE*RADIUS_BIN_SIZE*GRAD_BIN_SIZE_YZ
								+i*(RADIUS_BIN_SIZE*GRAD_BIN_SIZE_XZ)+j*GRAD_BIN_SIZE_XZ+k;
						
						data.put(0, idx, HistogramXZ[i][j][k]);
				
					}
					break;
				default:
					break;
				}
				
				
			}
		
	}
	
	
	private void fillHIST(CvMat[] NormMatArr)
	{
		double[] BinNumXY,BinNumXZ,BinNumYZ;
	
		BinNumXY= new double[4];
		BinNumYZ= new double[4];
		BinNumXZ= new double[4];
		
		for (int i = 0; i < MAT_ARRAY_SIZE; i++) {
			
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
			if((idx_ANGLE= getAngleBin(i))==-1)
				continue;
			
			if((idx_RADIUS= getRadiusBin(i))==-1)
				continue;
			
			
			HistogramXY[idx_ANGLE][idx_RADIUS][(int) BinNumXY[0]]+=weightsXY[0];
			HistogramXY[idx_ANGLE][idx_RADIUS][(int) BinNumXY[1]]+=weightsXY[1];
			
			HistogramYZ[idx_ANGLE][idx_RADIUS][(int) BinNumYZ[0]]+=weightsYZ[0];
			HistogramYZ[idx_ANGLE][idx_RADIUS][(int) BinNumYZ[1]]+=weightsYZ[1];
			
			HistogramXZ[idx_ANGLE][idx_RADIUS][(int) BinNumXZ[0]]+=weightsXZ[0];
			HistogramXZ[idx_ANGLE][idx_RADIUS][(int) BinNumXZ[1]]+=weightsXZ[1];
			
			
		}
				
		
	}
	
	private int getRadiusBin(int idx)
	{
		
		int x= idx%WIDTH;
		int y= idx/WIDTH;
		
		double distance=Math.sqrt(Math.pow(x-WIDTH/2,2)+Math.pow(y-HEIGHT/2,2));
		
		for (int i = 0; i < RADIUS_BIN_SIZE; i++) {
			
			if(RADIUS/RADIUS_BIN_SIZE*i<=distance && distance<RADIUS/RADIUS_BIN_SIZE *(i+1))
				return i;
		}	
		
		return -1;
	}
	
	private int getAngleBin(int idx)
	{
		
		int x= idx%WIDTH-WIDTH/2;
		int y= idx/WIDTH-HEIGHT/2;
		
		double Angle=0;
		
		if(( Angle=Math.atan2(y, x))<0)
			 Angle+=360;
		
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
		
		double[] weights= new double[2];
		
		weights[0]=Math.sin(theta[1])/(Math.sin(theta[0]+Math.sin(theta[1])));
		weights[1]=Math.sin(theta[0])/(Math.sin(theta[0]+Math.sin(theta[1])));
		
		
		return weights;
		
	}
	
	private double[] getBinNum(CvMat vector, int mode)
	{
		int binSize=0;
		double[] BinNum= new double[4];
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
		if((theta=Math.atan2(y, x))<0)
			theta+=360;
		
		
		///put bin number 
		for (int i = 0; i < binSize; i++) {
			
			if(FI/GRAD_BIN_SIZE_XY *i<=theta && theta<FI/GRAD_BIN_SIZE_XY*(i+1))
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
		
		///벡터 길이에따라서 다른 빈의 번호 저장
		
		
		
		return BinNum;
	}
	private void makeNormMatArr(short[] depthMap)
	{
		int imageHeight = HEIGHT;
		int imageWidth  = WIDTH;
		
		
		for (int i = 1; i < imageWidth-1; i++) {
			for (int j = 1; j < imageHeight-1; j++) {
				
				
				CvMat pt0 = cvCreateMat(3,1,CV_32FC1);
				CvMat pt1 = cvCreateMat(3,1,CV_32FC1);
				CvMat pt2 = cvCreateMat(3,1,CV_32FC1);
				CvMat pt3 = cvCreateMat(3,1,CV_32FC1);
				
//				double dataUP[]= {i,j-1,DepthMap[i+(j-1)*320]};
//				double dataDOWN[]= {i,j+1,DepthMap[i+(j+1)*320]};
//				double dataLEFT[]= {i-1,j,DepthMap[(i-1)+j*320]};
//				double dataRIGHT[]= {i+1,j,DepthMap[(i+1)+j*320]};
//					
//				pt0.put(dataUP);
//				pt1.put(dataDOWN);
//				pt2.put(dataLEFT);
//				pt3.put(dataRIGHT);
				
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
				
			
				
				CvMat PlaneNorm = getFittedPlaneNorm(pt0, pt1, pt2, pt3,0);
				
				NormMatArr[i+j*WIDTH]=PlaneNorm;
				
	
			}
			 
		}
	}
	
	private CvMat getFittedPlaneNorm(CvMat pt0,CvMat pt1, CvMat pt2,CvMat pt3,int mode)
	{
		
		CvMat v1 = cvCreateMat(3,1,CV_32FC1);
		CvMat v2 = cvCreateMat(3,1,CV_32FC1);
		CvMat normal= cvCreateMat(3, 1, CV_32FC1);
		
		
		for (int i = 0; i <3; i++) {
		
			if(mode==MANHATTAN_DISTANCE_MODE){
				
				v1.put(i, 0,Math.abs( pt0.get(i, 0)-pt1.get(i, 0)));
			
				v2.put(i, 0, Math.abs( pt2.get(i, 0)-pt3.get(i, 0)));
			}
		}
		cvCrossProduct(v1, v2, normal);
		
	
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
