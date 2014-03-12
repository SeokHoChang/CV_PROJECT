package depthPack;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import com.sun.corba.se.impl.encoding.CodeSetConversion.BTCConverter;

public class TestSetMaker {
	private static FileOutputStream fos=null;
	private static BufferedOutputStream bos=null;
	
	private static FileInputStream file=null;
	private static BufferedInputStream fis=null;
	private static int data_len=0;
	
	private static byte[] toBytes(Short input)
	{

		ByteBuffer buffer = ByteBuffer.allocate(2);
		buffer.putShort(input);
		buffer.flip();
		return buffer.array();
	}

	private static int bytesToShort(byte[] bytes) {

		 int newValue = 0;
         newValue |= (((int)bytes[0])<<8)&0xFF00;
         newValue |= (((int)bytes[1]))&0xFF;

         return newValue;
	}

	public static void createTestFile()
	{
		File f = new File("testSet.dat");
		if(!f.exists())
		{
			try {
				f.createNewFile();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	
	public static void recordReady()
	{
		try {
			fos = new FileOutputStream("testSet.dat");
			bos = new BufferedOutputStream(fos,320);

		int offset=0;
		bos.write(toBytes( (short) 320),offset,2);
		bos.write(toBytes( (short) 240),offset,2);

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	public static void recordFinish()
	{

		if(fos!=null)
		{
			try {
				fos.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	public static void recordTestFrameSet(short[] data) 
	{
	
		
		int offset=0;
		try {

			for(int i=0; i<76800;i++)
			{
//			System.out.println(data[i]+", "+offset);		
				bos.write(toBytes(data[i]),offset,2);

			}
			
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
	public static void loadReady()
	{

		byte[] temp= new byte[2];
		short[] data = null;
		int offset=0;
		try {
			file = new FileInputStream("testSet.dat");
			fis = new BufferedInputStream(file,320);
			
			fis.read(temp,offset,2);
			data_len=bytesToShort(temp);
			
			temp= new byte[2];
			fis.read(temp,offset,2);
			data_len*=bytesToShort(temp);
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	public static void loadFinish()
	{
		if(fis!=null)
		{
			try {
				fis.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
	public static short[] loadTestFrameSet()
	{
	
		byte[] temp= new byte[2];
		short[] data = null;
		int offset=0;
		try {
		
			data= new short[data_len];
			for(int i=0;i<data_len;i++)
			{
				temp= new byte[2];
				if(fis.read(temp,offset,2)==-1)
						return null;
					
				data[i]=(short) bytesToShort(temp);
				
			}
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
			return data;
			
		
	}
	
	
}
