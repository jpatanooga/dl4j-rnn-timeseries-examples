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
		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer( physioNetBaseDirectory, physioSchemaFilePath, physioLabelsFilePath );
		vec.loadSchema();
		vec.loadLabels();
		//vec.setupFileInputList(false, 0);
		vec.collectStatistics();
		
		
		// total dataset size == Smaller Class x 2
		
		int balancedMaxSize = vec.labels.diedLabelCount * 2;
		
		System.out.println( "> Balanced Dataset Max Size: " + balancedMaxSize );
		
		Map<String, String> negativeClassFiles = new LinkedHashMap<>();
		
		// survived
		Map<String, String> positiveClassFiles = new LinkedHashMap<>();
		
		//for (int x = 0; x < 100; x++) {
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
		
		File dstPath = new File( dstBasePath );
		
		System.out.println( "Copy to: " + dstPath );
		
		int posFilesCopied = 0;
		int negFilesCopied = 0;
		
		File[] listOfFiles = new File[ balancedMaxSize ];
		for ( int x = 0; x < balancedMaxSize; x++) {
			
			String patientID = "";
			if (positiveClassFiles.size() > negativeClassFiles.size()) {
				
				//positiveClassFiles.entrySet().
				
				//positiveClassFiles.
				Entry<String, String> entry = positiveClassFiles.entrySet().iterator().next();
				
				patientID = entry.getKey(); //positiveClassFiles.remove(0);
				positiveClassFiles.remove(patientID);
				
				String path = vec.srcDir + patientID + ".txt";
				listOfFiles[ x ] = new File( path );
				
		//		System.out.println( "pos: " + listOfFiles[ x ].getName() );
				
				posFilesCopied++;
				
			} else {
				
				Entry<String, String> entry = negativeClassFiles.entrySet().iterator().next();
				
				patientID = entry.getKey(); //positiveClassFiles.remove(0);
				negativeClassFiles.remove(patientID);
				
				
				//patientID = negativeClassFiles.remove(0);
				String path = vec.srcDir + patientID + ".txt";
				listOfFiles[ x ] = new File( path );
				
			//	System.out.println( "neg: " + path );
				negFilesCopied++;
				
			}
			
			
			FileUtils.copyFile( listOfFiles[ x ], new File(dstBasePath + listOfFiles[ x ].getName() ) );
			
			//filesCopied++;
			
			
		}		
		
		System.out.println( "Copied Files: " + (negFilesCopied + posFilesCopied) );
		System.out.println( "Positive Copied Files: " + ( posFilesCopied) );
		System.out.println( "Negative Copied Files: " + (negFilesCopied ) );
		
		
		
	}
	

}
