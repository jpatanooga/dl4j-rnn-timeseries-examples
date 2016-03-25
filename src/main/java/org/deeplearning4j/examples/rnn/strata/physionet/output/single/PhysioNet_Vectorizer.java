package org.deeplearning4j.examples.rnn.strata.physionet.output.single;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesDescriptorSchemaColumn;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class PhysioNet_Vectorizer extends org.deeplearning4j.examples.rnn.strata.physionet.PhysioNet_Vectorizer {

	public PhysioNet_Vectorizer(String srcDirectory, String schemaPath,
			String labels_file_path) {
		super(srcDirectory, schemaPath, labels_file_path);
		// TODO Auto-generated constructor stub
	}

	
	/**
	 * The Single output variant 
	 */
	@Override
	public DataSet generateNextTimeseriesVectorMiniBatch(int miniBatchSize, int currentMiniBatchOffset, int columnCount) {

		// minibatch size is the 1st dimension in the matrix, which we get as a parameter
		// do we have enough files left in our directory to give a full mini-batch? check for this

		//int columnCount = this.calculateTotalOutputColumnCount(); // 2nd dimension in matrix --- every column in the schema that is not !SKIP
		
		// what is the timestep count?
		
		int timestepCount = this.maxNumberTimeSteps; // 3rd dimension in matrix
		

		// RULE OF THUMB for features vs masks
		// features and labels are 3d
		// masks are always 2d
		
		INDArray input = Nd4j.zeros(new int[]{ miniBatchSize, columnCount, timestepCount } );
		// input mask should be 2d (no column count)
		//  we only care about minibatch and timestep --- for a given timestep, we are either using ALL colums... or we are not
		INDArray inputMask = Nd4j.zeros( new int[]{ miniBatchSize, timestepCount } );

		// have to make labels 3d, but we pad/mask everything but first timestep
		INDArray labels = Nd4j.zeros(new int[]{ miniBatchSize, 1, timestepCount } );
		// mask / pad everything in labels up to the LAST timestep? and put the real labels there
		INDArray labelsMask = Nd4j.zeros(new int[]{ miniBatchSize, timestepCount } ); // labels are always used
		
		int targetEndingIndex = miniBatchSize + currentMiniBatchOffset;
		
		int matrixMiniBatchTmpIndex = 0;
		
	    for (int fileIndex = currentMiniBatchOffset; fileIndex < targetEndingIndex; fileIndex++ ) {
	    		//this.listOfFilesToVectorize.length; i++) {

	    	if (this.listOfFilesToVectorize[ fileIndex ].isFile()) {
	    	
	    		// System.out.println("File: " + listOfFiles[i].getName() );
	    		
	    		String tmpPath = this.srcDir;
	    		if (tmpPath.trim().endsWith("/")) {
	    			
	    			tmpPath += this.listOfFilesToVectorize[ fileIndex ].getName();
	    			
	    		} else {
	    			
	    			tmpPath += "/" + this.listOfFilesToVectorize[ fileIndex ].getName();
	    			
	    		}
	    		
	    		//this.scanFileForStatistics( tmpPath );
	    		
	    	//	System.out.println( ">>" + fileIndex + " of " + targetEndingIndex + " -> " + tmpPath );
	    		this.extractFileContentsAndVectorize( tmpPath, matrixMiniBatchTmpIndex, columnCount, timestepCount, input, inputMask, labels, labelsMask );
	    		matrixMiniBatchTmpIndex++;
	    	
	    	} else if (this.listOfFilesToVectorize[ fileIndex ].isDirectory()) {
	    	
	    		//System.out.println("Directory: " + listOfFiles[i].getName());
	    	
	    	}
	    	
	    }		
		
		return new DataSet( input, labels, inputMask, labelsMask );
		
	}	
	
	
	
	
	
	@Override
	public void extractFileContentsAndVectorize(String filepath, int miniBatchIndex, int columnCount, int timeStepLength, INDArray dstInput, INDArray dstInputMask, INDArray dstLabels, INDArray dstLabelsMask) {
		
		
		
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
		    
		//    debugTreeMapData( timestampTreeMap );
		    
	
		    
		    
		    
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
		
	//	System.out.println("Debugging vectorization path ---------");
		
		//System.out.println( "length: " + dstInput.length() );
		//System.out.println( "cols: " + dstInput.columns() );

		
		int lastTimestamp = 0;
		int timeStepIndex = 0;
		
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
			
			int columnIndex = 0;

			// write the delta timestamp in the first column
			int[] params = new int[]{ miniBatchIndex, columnIndex, timeStepIndex };
		//	System.out.println( "Timestep Pass -------------------------- " );
		//	System.out.println( "Timestep: " + timeStepIndex );
//			System.out.println( "TS-Delta Params: " + miniBatchIndex + ", " + columnIndex + ", " + timeStepIndex );
			dstInput.putScalar(new int[]{ miniBatchIndex, columnIndex, timeStepIndex }, deltaT );
			dstInputMask.putScalar(new int[]{ miniBatchIndex, timeStepIndex }, 1.0 );
			columnIndex++;
			
			
			
//			System.out.print("[Delta-T]:" + deltaT +", Descriptors:[" );
			
			// first set of columns: Descriptors
			
			for (Map.Entry<String, TimeseriesDescriptorSchemaColumn> columnEntry : this.schema.getDescriptorColumnSchemas().entrySet()) {
				
				String key = columnEntry.getKey();
				TimeseriesDescriptorSchemaColumn schema_column = columnEntry.getValue();
				
				if (schema_column.transform == TimeseriesSchemaColumn.TransformType.SKIP) {
					
				} else {
				
					String val = generalDescriptorTreeMap.get( key );
					
					double transformedValue = schema_column.transformColumnValue( val );
//					System.out.println( "[" + key + ":" + val + " => " + transformedValue + "]" );
//					System.out.println( "Descriptor Params: " + miniBatchIndex + ", " + columnIndex + ", " + timeStepIndex );
					dstInput.putScalar(new int[]{ miniBatchIndex, columnIndex, timeStepIndex }, transformedValue );
					dstInputMask.putScalar(new int[]{ miniBatchIndex, timeStepIndex }, 1.0 );
					columnIndex++;
					
				}
				
				
			}
			
			
			
			
			
			
			
			//System.out.println( "]" );
//			System.out.println("[Delta-T]:" + deltaT );

			
			// now do the timeseries columns
			
			for (Map.Entry<String, TimeseriesSchemaColumn> columnEntry : this.schema.getTimeseriesColumnSchemas().entrySet()) {

				String key = columnEntry.getKey();
				TimeseriesSchemaColumn schema_column = columnEntry.getValue();
				
				if (schema_column.transform == TimeseriesSchemaColumn.TransformType.SKIP) {
				
					String val = valuesAtTimestamp.get(key);
//					System.out.println( "[" + key + ":" + val + " => SKIP]"  );
	
					
				} else {

						String val = valuesAtTimestamp.get(key);
						
						double transformedValue = schema_column.transformColumnValue( val );
						
//						System.out.println( "[" + key + ":" + val + " => " + transformedValue + "]" );
//						System.out.println( "TS Col Params: " + miniBatchIndex + ", " + columnIndex + ", " + timeStepIndex );
						params = new int[]{ miniBatchIndex, columnIndex, timeStepIndex };
						//System.out.println( "Timestep Pass -------------------------- " );
						//System.out.println( "Timestep: " + timeStepIndex );
						//System.out.println( "Current Params: " + params[0] + ", " + params[1] + ", " + params[2] );
						
						dstInput.putScalar( params, transformedValue );
						dstInputMask.putScalar(new int[]{ miniBatchIndex, timeStepIndex }, 1.0 );
						
						//dstInput.putScalar(new int[]{ miniBatchIndex, columnIndex, timeStepIndex }, transformedValue );
						columnIndex++;
						
						
				}
				
				
				
				
			}
			
//			System.out.println( "[end]" );
			
			// now put the vector into the input array for this timestep
			
			timeStepIndex++;
			
		} // for
		
		
		// now handle labels
		
		String patientID = generalDescriptorTreeMap.get("recordid");
		
	//	System.out.println( "Looking up label for patient ID: " + patientID );
		
		// what is the label? 1 -> survived, 0 -> died
		int labelOutcome = this.labels.translateLabelEntry(patientID);
		
		// for that index, put a 1
		
	//	System.out.println( "label indexes: " + miniBatchIndex + ", " + labelPositiveColumnIndex );
		
		int adjustedTimeStepIndex = timeStepIndex - 1;
		
	//	System.out.println( "timeStepIndex: " + timeStepIndex + ", adjustedTimeStepIndex: " + adjustedTimeStepIndex );
		
		// TODO: temp timestep for now, FIX THIS
		int[] label_params = new int[]{ miniBatchIndex, 0, adjustedTimeStepIndex };
		dstLabels.putScalar( label_params, labelOutcome );
		
		dstLabelsMask.putScalar( new int[]{ miniBatchIndex, adjustedTimeStepIndex }, 1 );
		
		
	}	
	
	
}
