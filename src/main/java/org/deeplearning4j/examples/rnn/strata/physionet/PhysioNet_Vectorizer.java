package org.deeplearning4j.examples.rnn.strata.physionet;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

import org.deeplearning4j.examples.rnn.strata.physionet.schema.PhysioNet_CSVSchema;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesDescriptorSchemaColumn;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;


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
	
	public int lastKnownTimestamp = 0;
	public int outOfOrderTimestampCount = 0;
	
	
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
	    
	    this.schema.computeDatasetStatistics();
	    
	    // now make the pass for derived statistics
	    System.out.println( "Scanning for derived Statistics ... " ); 
	    
	    for (int i = 0; i < listOfFiles.length; i++) {

	    	if (listOfFiles[i].isFile()) {
	    	
	    		// System.out.println("File: " + listOfFiles[i].getName() );
	    		
	    		String tmpPath = this.srcDir;
	    		if (tmpPath.trim().endsWith("/")) {
	    			
	    			tmpPath += listOfFiles[i].getName();
	    			
	    		} else {
	    			
	    			tmpPath += "/" + listOfFiles[i].getName();
	    			
	    		}
	    		
	    		this.scanFileForDerivedStatistics( tmpPath );
	    	
	    	} else if (listOfFiles[i].isDirectory()) {
	    	
	    		System.out.println("Directory: " + listOfFiles[i].getName());
	    	
	    	}
	    	
	    }	
	    
	    this.schema.computeDatasetDerivedStatistics();
	    
	    
		
	}
	
	public void scanFileForStatistics(String filepath) {
		
		String csvLine = "";
		
		try (BufferedReader br = new BufferedReader(new FileReader( filepath ) ) ) {
		    
		    
		    int descriptorLineCount = 0;
		    int timeseriesLineCount = 0;
		    Map<String, Integer> timeStepMap = new LinkedHashMap<>();
		    
		    while ((csvLine = br.readLine()) != null) {
		       // process the line.
		    	
				// open the file
		    	//String csvLine = value.toString();
		    	String[] columns = csvLine.split( columnDelimiter );
		    	


		    	
		    	
		    	//System.out.println( csvLine );
		    	if ( isRecordGeneralDescriptor(columns, this.schema) ) {
		    				    			
		    		
		    		 this.schema.evaluateInputRecord( csvLine );
		    		 descriptorLineCount++;
		    		
		    	} else if ( isHeader(columns) ) {
		    		
		    	//	System.out.println( "Skipping Header Line: " + csvLine );
		    		
		    	} else {
		    		
		    		this.schema.evaluateInputRecord( csvLine );
		    		timeseriesLineCount++;

		    		
		    		// now deal with a timeseries line
		    		
					String timeslot = columns[ 0 ].trim();
					
					int curTimeSlot = parseElapsedTimeForVisitInTotalMinutes( timeslot );
					
					if (this.lastKnownTimestamp > curTimeSlot) {
						// we just went back in time!
						this.outOfOrderTimestampCount++;
					}
					
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
			System.err.println( "" + csvLine );
		}		
		
		
		
		
	}
	
	public void scanFileForDerivedStatistics(String filepath) {
		
		
		
		try (BufferedReader br = new BufferedReader(new FileReader( filepath ) ) ) {
		    String csvLine;
		    
		    while ((csvLine = br.readLine()) != null) {
			       // process the line.
			    	
					// open the file
			    	//String csvLine = value.toString();
			    	//String[] columns = csvLine.split( columnDelimiter );
			    	
		    
		    
			    	this.schema.evaluateInputRecordForDerivedStatistics( csvLine );
		    
		    }
		    
		    
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
	    System.out.println( "Out of Order Timestamps: " + this.outOfOrderTimestampCount );
	    
	    
	    System.out.println( " ----------------- Vectorizer Process Stats ---------------- " );
		
		
	}
	
	/**
	 * Tells us if this is the first set of values at time offset "00:00"
	 * 
	 * @param line
	 * @return
	 */
	public static boolean isRecordGeneralDescriptor( String[] columns, PhysioNet_CSVSchema schema ) {
		
		String ts = columns[ 0 ].trim().toLowerCase();
		String colName = columns[ 1 ].trim().toLowerCase();
		
		//System.out.println( ts );
		
//		if ("weight".equals(colName) && "00:00".equals(ts)) {
//			return true;
//		}
		
		return schema.getDescriptorColumnSchemas().containsKey(colName) && "00:00".equals(ts);
		/*
		if (colVal.trim().equals("00:00")) {
			return true;
		}
		
		return false;
		*/
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
	public DataSet generateNextTimeseriesVectorMiniBatch(int miniBatchSize) {

		// minibatch size is the 1st dimension in the matrix, which we get as a parameter
		// do we have enough files left in our directory to give a full mini-batch? check for this

		int columnCount = this.calculateTotalOutputColumnCount(); // 2nd dimension in matrix --- every column in the schema that is not !SKIP
		
		// what is the timestep count?
		
		int timestepCount = this.maxNumberTimeSteps; // 3rd dimension in matrix
		
		
		INDArray input = Nd4j.zeros(new int[]{ miniBatchSize, columnCount, timestepCount } );
		INDArray labels = Nd4j.zeros(new int[]{ miniBatchSize, columnCount, timestepCount } );
		
		
		
		// for each mini-batch entry -> file
		for ( int m = 0; m < miniBatchSize; m++) {

			// open the file
			
			// for each (adjusted) timestep in the file, generate every column
			
			String filepath = "";
			
			this.extractFileContentsAndVectorize( filepath, columnCount, timestepCount, input, labels );
			
			
		}

		
		return new DataSet( input, labels );
		
	}
	
	public int calculateTotalOutputColumnCount() {
		
		// timestep delta-T
		
		// all columns we aren't skipping
		
		return 0;
		
	}
	
	
	/**
	 * Alex Black says...
	 * 
	 * 
	 * suggested down sampling to 202, and padding the lesser timeseries with 0's. 
		- also suggest replacing the timestamp value with "number of seconds from last timestep". 
		- this gets us out of the alignment issue that we could get into w less timesteps.
	 * 
	 * 
	 * 
	 * 
	 * Dave Kale says
	 * 
	 * We should just treat these as sequences of measurements, without worrying about time alignment. We have a sequence step whenever we have one or more measurements. We should not include any point in time where there are NO measurements.

		We will not have measurements of every variable at every sequence step. We should handle this by forward filling: carrying forward previous measurements, when available. When not available, we impute the median value of all measurements of that variable.
		
		Then we can incorporate time as an input to the RNN, in the form of either minutes elapsed since start or minutes elapsed since previous measurement, or both.
		
		That is how I suggest we begin. There are several variations to experiment with:
		
		1) Include a binary indicator for each variable, that equals 1 when we have an actual measurement and 0 when we've imputed. My money says this will help.
		
		2) Instead of imputation via forward-filling or median, we use zeros instead. I doubt this will work as well as the imputed values plus the binary indicators.
		
		3) Omit timestamps and consider only sequence order. I don't believe this will work as well, but there are folks in biomedical informatics who argue it will.

		Vectorization strategy:
		
		* multiple outputs, yes: in-hospital mortality, length of stay, survival in days. However, choosing form of those last two outputs could be tricky, since I think survival might be right censored or missing. Note that SAPS-I and SOFA are not outcomes but scores that we can treat as baseline predictors.
		
		* missing static data (e.g., weight): simplest thing for now is to impute median, maybe add a binary indicator
		
		* time series of length 0: here is what I propose for first pass: since we plan to integrate the "static" data into our RNN anyway, I propose that for episodes with no time series, we feed a single time step into the RNN that has all time series missing but passing in static data. This way, the model can predict from just static data.
		
		* rogue columns: those are typos. It seems clear to me that "TroponinI" and "TropininT" should be the TropI and TropT variables listed in the documentation.


	 * 
	 * Heuristic
	 * 
	 * 	-	we'll write directly into the dstInput 3dim array (matrix) and labels array
	 * 
	 * 
	 * Constraints
	 * 
	 * 	-	So we need the timestamp entries ordered (meaning we have to sort each file)
	 * 	-	we need the columns for a given timestamp in the same order as well, at every timestamp
	 * 
	 * 
	 * @param filepath
	 * @param columnCount
	 * @param timeStepLength
	 * @param dstInput
	 * @param dstLabels
	 */
	public void extractFileContentsAndVectorize(String filepath, int columnCount, int timeStepLength, INDArray dstInput, INDArray dstLabels) {
		
		Map<Integer, Map<String, String>> timestampTreeMap = new TreeMap< Integer, Map<String, String> >();
		Map<String, String> generalDescriptorTreeMap = new HashMap<>();
		
		// Pass 1: scan and sort all the timestamped entries --- we cant be sure they are ordered!
		try (BufferedReader br = new BufferedReader(new FileReader( filepath ) ) ) {
		    String csvLine;
		    
		    int descriptorLineCount = 0;
		    int timeseriesLineCount = 0;
		   // Map<String, Integer> timeStepMap = new LinkedHashMap<>();
		    
		    
		    while ((csvLine = br.readLine()) != null) {
		       // process the line.
		    	
				// open the file
		    	//String csvLine = value.toString();
		    	String[] columns = csvLine.split( columnDelimiter );
		    	
		    	
		    	if ( isRecordGeneralDescriptor(columns, this.schema) ) {
		    		
		    		 //this.schema.evaluateInputRecord( csvLine );
		    		 //descriptorLineCount++;
		    		
		    		generalDescriptorTreeMap.put( columns[1].trim().toLowerCase(), columns[2].trim() );
		    		
		    	} else if ( isHeader(columns) ) {
		    		
		    		
		    	} else {
		    		
		    		//this.schema.evaluateInputRecord( csvLine );
		    		timeseriesLineCount++;

		    		
		    		// now deal with a timeseries line
		    		
					String timeslot = columns[ 0 ].trim();
					
					// now for each timestep, order the data
					int curTimeSlot = parseElapsedTimeForVisitInTotalMinutes( timeslot );
					
					if (timestampTreeMap.containsKey(curTimeSlot)) {
						
						// already exists, add to it
						Map<String, String> tmpMap = timestampTreeMap.get(curTimeSlot);
						tmpMap.put( columns[ 1 ].trim().toLowerCase(), columns[ 2 ].trim() );
						
					} else {
						
						// add new one
						Map<String, String> tmpMap = new HashMap<String, String>();
						tmpMap.put( columns[ 1 ].trim().toLowerCase(), columns[ 2 ].trim() );
						timestampTreeMap.put( curTimeSlot, tmpMap );
						
					}
					
		    		
		    		
		    	}
		    	
		    }
		    
		    debugTreeMapData( timestampTreeMap );
		    
		    
		    
		    /*
		    if (timeStepMap.size() == 0) {
		    	
		    	System.out.println( "File " + filepath + " contained no timesteps!" );
		    	
		    }
		    */
		    
		    
		    
		    
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
		
		// Pass 2: vectorize the sortedMap Entries
		
		System.out.println("Debugging vectorization path ---------");
		int lastTimestamp = 0;
		
		//for ( int timeStep = 0; timeStep < timeStepLength; timeStep++ ) {
		for (Map.Entry<Integer, Map<String, String>> entry : timestampTreeMap.entrySet()) {
			  
	//		System.out.println(entry.getKey() + " => " + entry.getValue());
			int curTimestamp = entry.getKey();
			int deltaT = curTimestamp - lastTimestamp;
			
			lastTimestamp = curTimestamp;
			Map<String, String> valuesAtTimestamp = entry.getValue();

			
			// CONSIDER: DOES The file have data for the full 202 timesteps?
			// IF NOT >> need to do some padding
			
			// calculate delta-T for the adjusted timestamp column
			
			// this should match up to the schema -- scan the schema
//			for ( int col = 0; col < columnCount; col++ ) {
				
				// now run through every other column we want to drop in for this timestep
				
				
//			}
			
			
//			System.out.print("[Delta-T]:" + deltaT );
			
			// first set of columns: Descriptors
			
			for (Map.Entry<String, TimeseriesDescriptorSchemaColumn> columnEntry : this.schema.getDescriptorColumnSchemas().entrySet()) {
				
				String key = columnEntry.getKey();
				TimeseriesDescriptorSchemaColumn schema_column = columnEntry.getValue();
				
				String val = generalDescriptorTreeMap.get( key );
				
				double transformedValue = schema_column.transformColumnValue( val );
			/*	
				if (this.schema.customValueForMissingValue.equals( val )) {
					
					System.out.print( " ,[(d) " + key + ":MISSING]" );
					
				} else {
				*/
//					System.out.print( " ,[(d) " + key + ":" + val + " => " + transformedValue + "]" );
				//}
								
				
			}
			
			// now do the timeseries columns
			
			for (Map.Entry<String, TimeseriesSchemaColumn> columnEntry : this.schema.getTimeseriesColumnSchemas().entrySet()) {

				String key = columnEntry.getKey();
				TimeseriesSchemaColumn schema_column = columnEntry.getValue();
				
				if (schema_column.transform == TimeseriesSchemaColumn.TransformType.SKIP) {
				
					String val = valuesAtTimestamp.get(key);
//					System.out.print( " ,[ " + key + ":SKIP]" );
					
					
//				} else if (schema_column.columnTemporalType == TimeseriesSchemaColumn.ColumnTemporalType.DESCRIPTOR) {
					
					//String val = valuesAtTimestamp.get(key);

					
				} else {

					// does this column exist for this timestep?
				//	if (valuesAtTimestamp.containsKey(key) ) {
						
						String val = valuesAtTimestamp.get(key);
						
						double transformedValue = schema_column.transformColumnValue( val );
						
	//					System.out.print( " ,[(t) " + key + ": " + val + " => " + transformedValue + "]" );
						
				//	} else {
				//		System.out.print( " ,[(t) " + key + ":null" + "]" );
				//	}
					
				}
				
				
				
				
			}
			
	//		System.out.println( "[end]" );
			
			// now put the vector into the input array for this timestep
			
			
		}		
		
		
	}
	
	public static void debugTreeMapData(Map<Integer, Map<String, String>> timestampTreeMap) {
		
		for (Map.Entry<Integer, Map<String, String>> entry : timestampTreeMap.entrySet()) {
			  
			System.out.println(entry.getKey() + " => " + entry.getValue());
			
		}		
		
		
	}
	

	
	
	

}
