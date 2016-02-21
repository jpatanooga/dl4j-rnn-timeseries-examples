package org.deeplearning4j.examples.rnn.strata.physionet.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Calendar;

import org.deeplearning4j.examples.rnn.strata.physionet.PhysioNet_Vectorizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class PhysioNetVectorizationDebugTool {

	public static void triagePatientFile( String physioNetBaseDirectory, String physioSchemaFilePath, String physioLabelsFilePath, String patientFileFullPath ) {
		
		//PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("/tmp/set-a/", "src/test/resources/physionet_schema.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt" );
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer( physioNetBaseDirectory, physioSchemaFilePath, physioLabelsFilePath );
		vec.loadSchema();
		vec.loadLabels();
		
		vec.collectStatistics();
		
		String filename = patientFileFullPath; //vec.getFilenameForIndex( 0 );
//		if (filename.startsWith("/")) {
	//		filename = filename.replace("/", "");
		//}
		
		File f = new File(filename);
		String strippedFilename = f.getName().replace(".txt", "");
		
		BufferedWriter writer = null;
        try {
            //create a temporary file
            String timeLog = new SimpleDateFormat("yyyyMMdd_HHmmss").format(Calendar.getInstance().getTime());
            File logFile = new File("/tmp/" + strippedFilename + "_" + timeLog + ".txt");

            // This will output the full path where the file will be written to...
            System.out.println(logFile.getCanonicalPath());
            writer = new BufferedWriter(new FileWriter(logFile));
		
	
			// log this
			vec.schema.debugPrintDatasetStatistics();
			vec.schema.logDatasetStatistics(writer);
			vec.schema.logColumns(writer);
			
			// now write out the raw record data
			
			//System.out.println( "ND4J Input Size: " );
			//System.out.println( "Minibatch: 1" );
			writer.write( "\n\n" );
			writer.write( "Total Column Count: " + (vec.schema.getTransformedVectorSize() + 1) + "\n" );
			writer.write( "Total Timestep Count: " + vec.maxNumberTimeSteps + "\n" );
			writer.write( "\n\n" );
			
			vec.logTreeMapData(writer, filename);
			
			int timesteps = vec.maxNumberTimeSteps;
			
			INDArray input = Nd4j.zeros(new int[]{ 1, vec.schema.getTransformedVectorSize() + 1, timesteps });
			INDArray inputMask = Nd4j.zeros(new int[]{ 1, vec.schema.getTransformedVectorSize() + 1, timesteps });
			// 1 == mini-batch size
			// 2 == number of classes (0 -> no survive, 1 -> survival)
			INDArray labels = Nd4j.zeros(new int[]{ 1, 2 });
			INDArray labelsMask = Nd4j.ones(new int[]{ 1, 2 });
			
			vec.extractFileContentsAndVectorize( filename, 0, vec.schema.getTransformedVectorSize() + 1, timesteps, input, inputMask, labels, labelsMask);
			
			PhysioNet_Vectorizer.log_debug3D_Nd4J_Input( writer, input, 1, vec.schema.getTransformedVectorSize() + 1, timesteps );
			
			writer.write( "\n\nDebug Input Mask --------------- \n" );
			
			PhysioNet_Vectorizer.log_debug3D_Nd4J_Input( writer, inputMask, 1, vec.schema.getTransformedVectorSize() + 1, timesteps );
	
			//PhysioNet_Vectorizer.log_debug3D_Nd4J_Input( writer, d.getFeaturesMaskArray(), 1, 43, 202 );
			
			System.out.println( "> [ done ] ");

	    } catch (Exception e) {
	        e.printStackTrace();
	    } finally {
	        try {
	            // Close the writer regardless of what happens...
	            writer.close();
	        } catch (Exception e) {
	        }
	    }		
		
				
		
		
	}	
	
}
