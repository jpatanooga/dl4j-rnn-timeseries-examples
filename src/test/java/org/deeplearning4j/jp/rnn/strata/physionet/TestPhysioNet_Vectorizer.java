package org.deeplearning4j.jp.rnn.strata.physionet;

import static org.junit.Assert.*;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Map;
import java.util.TreeMap;

import org.deeplearning4j.examples.rnn.strata.physionet.PhysioNetLabels;
import org.deeplearning4j.examples.rnn.strata.physionet.PhysioNet_Vectorizer;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesDescriptorSchemaColumn;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn;
import org.deeplearning4j.examples.rnn.strata.physionet.utils.PhysioNetVectorizationDebugTool;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestPhysioNet_Vectorizer {

	// need to auto-download:
	// https://www.physionet.org/challenge/2012/Outcomes-a.txt

	@Test
	public void testScanDirectory() {
		
		//PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("src/test/resources/data/physionet/sample/set-a/", "src/test/resources/physionet_schema.txt" );
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("/tmp/set-a/", "src/test/resources/physionet_schema.txt", "src/test/resources/sample/set-a-label/Outcomes-a.txt" );
		vec.loadSchema();

		vec.collectStatistics();
		vec.schema.computeDatasetStatistics();
		
		vec.debugStats();
		
		vec.schema.debugPrintRogueColumns();

		vec.schema.debugPrintDatasetStatistics();

	}

	@Test
	public void testScanAlternateStatisticsDirectory() {
		
		//PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("src/test/resources/data/physionet/sample/set-a/", "src/test/resources/physionet_schema.txt" );
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("/tmp/set-a-balanced-5/train/", "src/test/resources/physionet_schema.txt", "src/test/resources/sample/set-a-label/Outcomes-a.txt" );
		vec.loadSchema();

		vec.setSpecialStatisticsFileList("/tmp/set-a/");
		
		vec.collectStatistics();
		vec.schema.computeDatasetStatistics();
		
		vec.debugStats();
		
		vec.schema.debugPrintRogueColumns();

		vec.schema.debugPrintDatasetStatistics();

	}
	
	
	@Test
	public void testLabelParsing() {
		
		
		PhysioNetLabels labels = new PhysioNetLabels();
		labels.load( "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt" );
		
    	// Cols per patient: { survival_days, length_stay_days, in_hospital_death }
		
		Map<String, Integer> patientMap = labels.getPatientData( "132567" );
		assertEquals( 7, patientMap.get("length_stay_days").intValue() );
		assertEquals( -1, patientMap.get("survival_days").intValue() );
		assertEquals( 0, patientMap.get("in_hospital_death").intValue() );
		
		assertEquals( 1, labels.translateLabelEntry("132567") );
		
		assertEquals( 0, labels.translateLabelEntry("132811") );
		
		System.out.println( "Label for 135361: " + labels.translateLabelEntry("135361") );
		
	}
	
	@Test
	public void testSchemaLoad() {
		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("src/test/resources/", "src/test/resources/physionet_schema.txt", "src/test/resources/sample/set-a-label/Outcomes-a.txt" );
		//vec.collectStatistics();
		vec.loadSchema();
		vec.schema.debugPrintDatasetStatistics();
		
	}
	
	
	
	@Test
	public void testSkipHeader() {

		Map<Integer, String> map = new TreeMap<Integer, String>();

		// Add Items to the TreeMap
		map.put(new Integer(11), "One");
		map.put(new Integer(2), "Two");
		map.put(new Integer(3), "Three");

		// Iterate over them
		for(Map.Entry<Integer,String> entry : map.entrySet()) {
		  System.out.println(entry.getKey() + " => " + entry.getValue());
		}		
		
	}
	
	@Test
	public void testVectorizeFile() {
		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("/tmp/set-a/", "src/test/resources/physionet_schema.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt" );
		vec.loadSchema();
		vec.loadLabels();

		vec.collectStatistics();
		vec.schema.computeDatasetStatistics();

		//vec.extractFileContentsAndVectorize( "/tmp/set-a/140505.txt", 1, 1, null, null);
		
		vec.schema.debugPrintDatasetStatistics();
		
	}

	@Test
	public void testVectorizeFile_SchemaTransform() {
		
/*
   @ATTRIBUTE icutype  		NUMERIC DESCRIPTOR   !NORMALIZE !REPLACE=5
   @ATTRIBUTE albumin		NUMERIC TIMESERIES		!NORMALIZE !PAD_TAIL_WITH_ZEROS
   @ATTRIBUTE alp		NUMERIC TIMESERIES		!NORMALIZE !PAD_TAIL_WITH_ZEROS
		
 */
		
		

		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("/tmp/set-a/", "src/test/resources/physionet_schema.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt" );
		vec.loadSchema();
		vec.loadLabels();
		
		assertEquals( "-1", vec.schema.customValueForMissingValue );
		
		TimeseriesDescriptorSchemaColumn column_icutype = vec.schema.getDescriptorColumnSchemaByName("icutype");
		assertEquals( TimeseriesSchemaColumn.ColumnDescriptorMissingValueStrategy.ZERO, column_icutype.missingValStrategy );
		//assertEquals( "5", column_icutype.customMissingValueReplacementValue );

		TimeseriesSchemaColumn column_platelets = vec.schema.getTimeseriesColumnSchemaByName("alp");
		assertEquals( TimeseriesSchemaColumn.ColumnTimeseriesPaddingStrategyType.PAD_TAIL_WITH_ZEROS, column_platelets.paddingStrategy );
		//assertEquals( "0", column_platelets.customMissingValueReplacementValue );
		
		
		
		
		vec.collectStatistics();
		//vec.schema.computeDatasetStatistics();

	//	vec.schema.debugPrintDatasetStatistics();		
		vec.schema.debugPrintDatasetStatistics();
		
		System.out.println( "ND4J Input Size: " );
		System.out.println( "Minibatch: 1" );
		System.out.println( "Column Count: " + (vec.schema.getTransformedVectorSize() + 1) );
		System.out.println( "Timestep Count: " + 2 );
		
		int timesteps = 3;
		
		INDArray input = Nd4j.zeros(new int[]{ 1, vec.schema.getTransformedVectorSize() + 1, timesteps });
		INDArray inputMask = Nd4j.zeros(new int[]{ 1, vec.schema.getTransformedVectorSize() + 1, timesteps });
		// 1 == mini-batch size
		// 2 == number of classes (0 -> no survive, 1 -> survival)
		INDArray labels = Nd4j.zeros(new int[]{ 1, 2 });
		INDArray labelsMask = Nd4j.ones(new int[]{ 1, 2 });
		
		vec.extractFileContentsAndVectorize( "src/test/resources/physionet_sample_data.txt", 0, vec.schema.getTransformedVectorSize() + 1, timesteps, input, inputMask, labels, labelsMask);
		
		PhysioNet_Vectorizer.debug3D_Nd4J_Input( input, 1, vec.schema.getTransformedVectorSize() + 1, timesteps );
		
		System.out.println( "Debug Input Mask --------------- " );
		
		PhysioNet_Vectorizer.debug3D_Nd4J_Input( inputMask, 1, vec.schema.getTransformedVectorSize() + 1, timesteps );

		
		double labelNegativeLabelValue = labels.getDouble(0, 0);
		double labelPositiveLabelValue = labels.getDouble(0, 1);
		assertEquals( 0.0, labelNegativeLabelValue, 0.0 );
		assertEquals( 1.0, labelPositiveLabelValue, 0.0 );
		
	}

	
	@Test
	public void testParseGeneralDescriptorValues() {
		
		String csvLine = "00:00,RecordID,135361";
		String[] columns = csvLine.split( "," );
		
		//PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("src/test/resources/", "src/test/resources/physionet_schema.txt", "src/test/resources/sample/set-a-label/Outcomes-a.txt" );
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("/tmp/set-a/", "src/test/resources/physionet_schema.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt" );
		vec.loadSchema();
		//vec.schema.debugPrintDatasetStatistics();
		
		assertEquals( true, vec.isRecordGeneralDescriptor(columns, vec.schema) );
		
	}

	@Test
	public void testParseTimeseriesValues() {

		String testValue_0 = "00:00";
		String testValue_1 = "00:07";
		String testValue_2 = "26:07";
		
		int result_0 = PhysioNet_Vectorizer.parseElapsedTimeForVisitInTotalMinutes(testValue_0);
		assertEquals( 0, result_0 );

		int result_1 = PhysioNet_Vectorizer.parseElapsedTimeForVisitInTotalMinutes(testValue_1);
		assertEquals( 7, result_1 );

		int result_2 = PhysioNet_Vectorizer.parseElapsedTimeForVisitInTotalMinutes(testValue_2);
		assertEquals( 1567, result_2 );

		
	}

	@Test
	public void testSingleFileVectorizeAndLog() {

		String physioNetBaseDirectory = "/tmp/set-a/";
		String physioSchemaFilePath = "src/test/resources/physionet_schema.txt";
		String physioLabelsFilePath = "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt";
		String patientFileFullPath = "/tmp/set-a/132959.txt";
		
		PhysioNetVectorizationDebugTool.triagePatientFile(physioNetBaseDirectory, physioSchemaFilePath, physioLabelsFilePath, patientFileFullPath);
		
		/*
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("/tmp/set-a/", "src/test/resources/physionet_schema.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt" );
		vec.loadSchema();
		vec.loadLabels();
		
		vec.collectStatistics();
		
		String filename = "/tmp/set-a/132959.txt"; //vec.getFilenameForIndex( 0 );
//		if (filename.startsWith("/")) {
	//		filename = filename.replace("/", "");
		//}
		
		File f = new File(filename);
		String strippedFilename = f.getName();
		
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
			
			System.out.println( "Debug Input Mask --------------- " );
			
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
		*/
				
		
		
	}
	
	
}
