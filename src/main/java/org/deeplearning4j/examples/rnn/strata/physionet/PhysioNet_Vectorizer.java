package org.deeplearning4j.examples.rnn.strata.physionet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import org.deeplearning4j.examples.rnn.strata.physionet.schema.PhysioNet_CSVSchema;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn;

/**
 * Scans over timeseries dataset and collects statistics.
 * 
 * Serves as custom schema system combined with vectorizer
 * 
 * 
 * @author josh
 *
 */
public class PhysioNet_Vectorizer {
	
	boolean hasCollectedStatistics = false;
	String srcDir = null;
	String currentPatientFile = null;
	String columnDelimiter = ",";
	String schemaPath = "";
	
	public PhysioNet_CSVSchema schema = null;
	
	public int minNumberTimeseriesEntriesForPatientRecord = 100; // think better about this
	public int maxNumberTimeseriesEntriesForPatientRecord = 0;

	public int minNumberTimeSteps = 100;
	public int maxNumberTimeSteps = 0;
	
	
	public PhysioNet_Vectorizer(String srcDirectory, String schemaPath) {
		
		this.srcDir = srcDirectory;
		this.schema = new PhysioNet_CSVSchema( );
		this.schemaPath = schemaPath;
		
	}
	
	public void loadSchema() {
		
		try {
			this.schema.parseSchemaFile( this.schemaPath );
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	/**
	 * The purpose of collecting summary statistics for PhysioNet is to 
	 * 
	 * 	1. discover all of the columns
	 * 	2. get the ranges of their values
	 * 
	 */
	public void collectStatistics() {
		
		// for each patient file
		
		System.out.println( "scanning: " + this.srcDir );
		
		File folder = new File( this.srcDir );
		
		if (!folder.exists()) {
			System.out.println("File Does Not Exist.");
			return;
		}
		
		if (folder.isDirectory()) {
			
		} else {
			System.out.println("This is a single file");
		}
		
		File[] listOfFiles = folder.listFiles();
		
		 System.out.println( "Found Files: " + listOfFiles.length + "\n" );

	    for (int i = 0; i < listOfFiles.length; i++) {

	    	if (listOfFiles[i].isFile()) {
	    	
	    		// System.out.println("File: " + listOfFiles[i].getName() );
	    		
	    		String tmpPath = this.srcDir;
	    		if (tmpPath.trim().endsWith("/")) {
	    			
	    			tmpPath += listOfFiles[i].getName();
	    			
	    		} else {
	    			
	    			tmpPath += "/" + listOfFiles[i].getName();
	    			
	    		}
	    		
	    		this.scanFileForStatistics( tmpPath );
	    	
	    	} else if (listOfFiles[i].isDirectory()) {
	    	
	    		System.out.println("Directory: " + listOfFiles[i].getName());
	    	
	    	}
	    	
	    }
	    
	    
		
	}
	
	public void scanFileForStatistics(String filepath) {
		
		try (BufferedReader br = new BufferedReader(new FileReader( filepath ) ) ) {
		    String csvLine;
		    
		    int descriptorLineCount = 0;
		    int timeseriesLineCount = 0;
		    Map<String, Integer> timeStepMap = new LinkedHashMap<>();
		    
		    while ((csvLine = br.readLine()) != null) {
		       // process the line.
		    	
				// open the file
		    	//String csvLine = value.toString();
		    	String[] columns = csvLine.split( columnDelimiter );
		    	


		    	
		    	
		    	//System.out.println( csvLine );
		    	if ( isRecordGeneralDescriptor(columns) ) {
		    		
		    		 this.schema.evaluateInputRecord( csvLine );
		    		 descriptorLineCount++;
		    		
		    	} else if ( isHeader(columns) ) {
		    		
		    	//	System.out.println( "Skipping Header Line: " + csvLine );
		    		
		    	} else {
		    		
		    		this.schema.evaluateInputRecord( csvLine );
		    		timeseriesLineCount++;

		    		
		    		// now deal with a timeseries line
		    		
					String timeslot = columns[ 0 ].trim();
					
					if (timeStepMap.containsKey(timeslot)) {
						 // increment
						
					} else {
						// add key
						timeStepMap.put(timeslot, 1);
					}
		    		
		    		
		    	}
		    	
		    }
		    
		    if (timeStepMap.size() == 0) {
		    	
		    	System.out.println( "File " + filepath + " contained no timesteps!" );
		    	
		    }
		    
		    //System.out.println( "Stats for: " + filepath );
		    //System.out.println( "Descriptor Lines: " + descriptorLineCount );
		    //System.out.println( "Timeseries Lines: " + timeseriesLineCount );
		    
		    if (timeseriesLineCount > this.maxNumberTimeseriesEntriesForPatientRecord) {
		    	this.maxNumberTimeseriesEntriesForPatientRecord = timeseriesLineCount;
		    }

		    if (timeseriesLineCount < this.minNumberTimeseriesEntriesForPatientRecord) {
		    	this.minNumberTimeseriesEntriesForPatientRecord = timeseriesLineCount;
		    }
		    
		    if ( timeStepMap.size() > this.maxNumberTimeSteps) {
		    	this.maxNumberTimeSteps = timeStepMap.size();
		    }
		    
		    if ( timeStepMap.size() < this.minNumberTimeSteps) {
		    	this.minNumberTimeSteps = timeStepMap.size();
		    }
		    
		    
		    //System.out.println( "Min Timeseries In a Record So Far: " + this.minNumberTimeseriesEntriesForPatientRecord );
		    //System.out.println( "Max Timeseries In a Record So Far: " + this.maxNumberTimeseriesEntriesForPatientRecord );
		    
		    
		    
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
		
		
		
		
	}
	
	public void debugStats() {
		
		System.out.println( " ----------------- Vectorizer Process Stats ---------------- " );
	    System.out.println( "Min Timeseries In a Record: " + this.minNumberTimeseriesEntriesForPatientRecord );
	    System.out.println( "Max Timeseries In a Record: " + this.maxNumberTimeseriesEntriesForPatientRecord );
	    
	    System.out.println( "Min TimeSteps In a Record: " + this.minNumberTimeSteps );
	    System.out.println( "Max TimeSteps In a Record: " + this.maxNumberTimeSteps );
	    
	    System.out.println( " ----------------- Vectorizer Process Stats ---------------- " );
		
		
	}
	
	/**
	 * Tells us if this is the first set of values at time offset "00:00"
	 * 
	 * @param line
	 * @return
	 */
	public static boolean isRecordGeneralDescriptor( String[] columns ) {
		
		String colVal = columns[ 0 ];
		
		if (colVal.trim().equals("00:00")) {
			return true;
		}
		
		return false;
	}
	
	public static boolean isHeader( String[] columns ) {
		
		String colVal = columns[ 0 ];
		
		if (colVal.trim().equals("Time")) {
			return true;
		}
		
		return false;
	}
	
	
	
	public static int parseElapsedTimeForVisitInTotalMinutes(String timeFormatRaw) {
		
		String[] parts = timeFormatRaw.trim().split(":");
		String hours = parts[ 0 ];
		String minutes = parts[ 1 ];
		
		int iHours = Integer.parseInt( hours );
		int iMinutes = Integer.parseInt( minutes );
		
		
		
		return (60 * iHours) + iMinutes;
	}
	
	/**
	 * Mini-batch size: 
	 * 		number of patients in a batch
	 * 		also: number of files in the batch to open (we open a file and generate a slice of the mini-batch 3d output) 
	 * 
	 * 
	 * TODO:
	 * 		-	how do we handle labels?
	 * 
	 */
	public void generateNextTimeseriesVectorMiniBatch(int miniBatchSize) {
		
		// minibatch size is the 1st dimension in the matrix, which we get as a parameter
		// do we have enough files left in our directory to give a full mini-batch? check for this

		int columnCount = this.calculateTotalOutputColumnCount(); // 2nd dimension in matrix --- every column in the schema that is not !SKIP
		
		// what is the timestep count?
		
		int timestepCount = this.maxNumberTimeSteps; // 3rd dimension in matrix
		
		// for each mini-batch entry -> file
		for ( int m = 0; m < miniBatchSize; m++) {

			// open the file
			
			// for each (adjusted) timestep in the file, generate every column
			
			for ( int timeStep = 0; timeStep < timestepCount; timeStep++ ) {
				
				// CONSIDER: DOES The file have data for the full 202 timesteps?
				// IF NOT >> need to do some padding
				
				// calculate delta-T for the adjusted timestamp column
				
				for ( int col = 0; col < columnCount; col++ ) {
					
					// now run through every other column we want to drop in for this timestep
					
					
				}
				
			}
			
		}

		
		
	}
	
	public int calculateTotalOutputColumnCount() {
		
		// timestep delta-T
		
		// all columns we aren't skipping
		
		return 0;
		
	}
	
	

}
