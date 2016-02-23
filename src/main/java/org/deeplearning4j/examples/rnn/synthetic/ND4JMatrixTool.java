package org.deeplearning4j.examples.rnn.synthetic;

import java.io.BufferedWriter;
import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;

public class ND4JMatrixTool {

	public static void debug3D_Nd4J_Input( INDArray dstInput, int miniBatchCount, int columnCount, int timeStepCount) {
		
		System.out.println( "Debugging Input of ND4J 3d Matrix -------" );
		
		for ( int miniBatchIndex = 0; miniBatchIndex < miniBatchCount; miniBatchIndex++) {
		
			System.out.println( "Mini-Batch Index: " + miniBatchIndex );
			
			for ( int timeStepIndex = 0; timeStepIndex < timeStepCount; timeStepIndex++) {
				
				System.out.print( "[timestep: " + timeStepIndex + "] " );
				
				for ( int columnIndex = 0; columnIndex < columnCount; columnIndex++) {
				
				

					int[] params = new int[]{ miniBatchIndex, columnIndex, timeStepIndex };
					
					double v = dstInput.getDouble( params );
					
					System.out.print( ", " + v );
					
					
				}
				
				System.out.println("");
				
			}
			
		
		}
		
		System.out.println( "Debugging Input of ND4J 3d Matrix -------" );
		
	}	
	
	public static void debug2D_Nd4J_Input( INDArray dstInput, int miniBatchCount, int timeStepCount) {
		
		System.out.println( "\nSTART > Debugging Input of ND4J 2d Matrix -------" );
		
		for ( int miniBatchIndex = 0; miniBatchIndex < miniBatchCount; miniBatchIndex++) {
		
			System.out.println( "Mini-Batch Index: " + miniBatchIndex );
			
			for ( int timeStepIndex = 0; timeStepIndex < timeStepCount; timeStepIndex++) {
				
				System.out.print( "[timestep: " + timeStepIndex + "] " );
				
			

				int[] params = new int[]{ miniBatchIndex, timeStepIndex };
				
				double v = dstInput.getDouble( params );
				
				System.out.print( ", " + v );
				
				
				System.out.println("");
				
			}
			
		
		}
		
		System.out.println( "END > Debugging Input of ND4J 2d Matrix -------" );
		
	}		
	
	
	public static void log_debug3D_Nd4J_Input( BufferedWriter writer, INDArray dstInput, int miniBatchCount, int columnCount, int timeStepCount) throws IOException {
		
		writer.write( "Debugging Input of ND4J 3d Matrix -------\n" );
		
		for ( int miniBatchIndex = 0; miniBatchIndex < miniBatchCount; miniBatchIndex++) {
		
			writer.write( "Mini-Batch Index: " + miniBatchIndex + "\n" );
			
			for ( int timeStepIndex = 0; timeStepIndex < timeStepCount; timeStepIndex++) {
				
				writer.write( "[timestep: " + timeStepIndex + "] " );
				
				for ( int columnIndex = 0; columnIndex < columnCount; columnIndex++) {
				
				

					int[] params = new int[]{ miniBatchIndex, columnIndex, timeStepIndex };
					
					double v = dstInput.getDouble( params );
					
					writer.write( ", " + v );
					
					
				}
				
				writer.write("\n");
				
			}
			
		
		}
		
		writer.write( "END Debugging Input of ND4J 3d Matrix -------\n" );
		
	}		
	
}
