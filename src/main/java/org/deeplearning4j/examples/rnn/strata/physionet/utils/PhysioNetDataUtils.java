package org.deeplearning4j.examples.rnn.strata.physionet.utils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.examples.rnn.strata.physionet.PhysioNet_Vectorizer;

public class PhysioNetDataUtils {
	
	
	
	public static void extractNFoldSubset(boolean balanced, String srcDirectory, String schemaPath, String labels_file_path, String dstBasePath) throws IOException {
		
		String physioNetBaseDirectory = "/tmp/set-a/";
		String physioSchemaFilePath = "src/test/resources/physionet_schema.txt";
		String physioLabelsFilePath = "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt";
		
		String dstPathTrainingDirectory = dstBasePath + "train/";
		String dstPathTestDirectory = dstBasePath + "test/";
		
		int testDatasetSize = 200;
		
		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer( physioNetBaseDirectory, physioSchemaFilePath, physioLabelsFilePath );
		vec.loadSchema();
		vec.loadLabels();
		//vec.setupFileInputList(false, 0);
		vec.collectStatistics();
		
		
		// total dataset size == Smaller Class x 2
		
		int balancedMaxSize = vec.labels.diedLabelCount * 2;
		
		int trainDatasetSize = balancedMaxSize - testDatasetSize;
		
		//System.out.println( "> Balanced Dataset Max Size: " + balancedMaxSize );
		System.out.println( "> Balanced [Train] Dataset Size: " + trainDatasetSize );
		System.out.println( "> Balanced [Test] Dataset Size: " + testDatasetSize );
		
		Map<String, String> negativeClassFiles = new LinkedHashMap<>();
		Map<String, String> negativeClassFilesTest = new LinkedHashMap<>();
		
		// survived
		Map<String, String> positiveClassFiles = new LinkedHashMap<>();
		Map<String, String> positiveClassFilesTest = new LinkedHashMap<>();
		
		// get all of the examples we can up to the max
		while (negativeClassFiles.size() + positiveClassFiles.size() < balancedMaxSize) {
			
			int randomIndex = (int )(Math.random() * 4000);
			
			String filename = vec.getFilenameForIndex( randomIndex );
			
			String[] filenameParts = filename.split(".t");
			String patientID = filenameParts[ 0 ];
			
			//System.out.println( "" + patientID );

			
			
			if (0 == vec.labels.translateLabelEntry(patientID)) {
				
				if (negativeClassFiles.containsKey(patientID)) {
					
				} else {
				
					// died
					if (negativeClassFiles.size() < balancedMaxSize / 2) { 
						negativeClassFiles.put( patientID, "" );
					}
					
				}
				
			} else {
				
				if (positiveClassFiles.containsKey(patientID)) {
					
				} else {
				
					
					// survived
					if (positiveClassFiles.size() < balancedMaxSize / 2) {
						positiveClassFiles.put( patientID, "" );
					}
					
				}
				
			}
			
		} // while
		
		System.out.println( "Subset Setup -> Classes: " );
		
		System.out.println( "Survived: " + positiveClassFiles.size() );
		System.out.println( "Died: " + negativeClassFiles.size() );
		
		
		// NOW bleed off the test subset
		
		//File dstPathTest = new File( dstPathTestDirectory );
		
		System.out.println( "Copy Test Files To: " + dstPathTestDirectory );
		
		
		int loopMax = testDatasetSize / 2;
		for (int x = 0; x < loopMax; x++ ) {
			
			String patientID = "";
			
			Entry<String, String> posEntry = positiveClassFiles.entrySet().iterator().next();
			patientID = posEntry.getKey(); 
			positiveClassFiles.remove(patientID);
			
			//positiveClassFilesTest.put(patientID, "");
			
			// just copy now
			String path = vec.srcDir + patientID + ".txt";
			
			FileUtils.copyFile( new File( path ), new File(dstPathTestDirectory + patientID + ".txt" ) );
			
			

			Entry<String, String> negEntry = negativeClassFiles.entrySet().iterator().next();
			patientID = negEntry.getKey(); 
			negativeClassFiles.remove(patientID);
			
			//negativeClassFilesTest.put(patientID, "");
			
			// just copy now
			
			path = vec.srcDir + patientID + ".txt";
			
			FileUtils.copyFile( new File( path ), new File(dstPathTestDirectory + patientID + ".txt" ) );
			
			
		}
		
		
		
		
		//File dstPathTraining = new File( dstPathTrainingDirectory );
		
		System.out.println( "Copy Training Files To: " + dstPathTrainingDirectory );
		
		int posFilesCopied = 0;
		int negFilesCopied = 0;
		
		if ( trainDatasetSize != positiveClassFiles.size() + negativeClassFiles.size() ) {
			System.err.println( "ERR: invalid copy size for remaining training data..." );
		}
		
		//File[] listOfFiles = new File[ balancedMaxSize ];
		for ( int x = 0; x < trainDatasetSize; x++) {
			
			String patientID = "";
			if (positiveClassFiles.size() > negativeClassFiles.size()) {
				
				//positiveClassFiles.entrySet().
				
				//positiveClassFiles.
				Entry<String, String> entry = positiveClassFiles.entrySet().iterator().next();
				
				patientID = entry.getKey(); //positiveClassFiles.remove(0);
				positiveClassFiles.remove(patientID);
				
				String path = vec.srcDir + patientID + ".txt";
				//listOfFiles[ x ] = new File( path );
				
		//		System.out.println( "pos: " + listOfFiles[ x ].getName() );
				
				FileUtils.copyFile( new File(path), new File(dstPathTrainingDirectory + patientID + ".txt" ) );
				
				posFilesCopied++;
				
			} else {
				
				Entry<String, String> entry = negativeClassFiles.entrySet().iterator().next();
				
				patientID = entry.getKey(); //positiveClassFiles.remove(0);
				negativeClassFiles.remove(patientID);
				
				
				//patientID = negativeClassFiles.remove(0);
				String path = vec.srcDir + patientID + ".txt";
				//listOfFiles[ x ] = new File( path );
				
				FileUtils.copyFile( new File(path), new File(dstPathTrainingDirectory + patientID + ".txt" ) );
				
			//	System.out.println( "neg: " + path );
				negFilesCopied++;
				
			}
			
			
			
			
			//filesCopied++;
			
			
		}		
		
		System.out.println( "Copied Files: " + (negFilesCopied + posFilesCopied) );
		System.out.println( "Positive Copied Files: " + ( posFilesCopied) );
		System.out.println( "Negative Copied Files: " + (negFilesCopied ) );
		
		
		
	}
	

}
