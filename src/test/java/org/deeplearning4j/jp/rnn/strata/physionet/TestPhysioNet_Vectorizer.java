package org.deeplearning4j.jp.rnn.strata.physionet;

import static org.junit.Assert.*;

import java.util.Map;
import java.util.TreeMap;

import org.deeplearning4j.examples.rnn.strata.physionet.PhysioNet_Vectorizer;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesDescriptorSchemaColumn;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestPhysioNet_Vectorizer {


	@Test
	public void testScanDirectory() {
		
		//PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("src/test/resources/data/physionet/sample/set-a/", "src/test/resources/physionet_schema.txt" );
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("/tmp/set-a/", "src/test/resources/physionet_schema.txt" );
		vec.loadSchema();

		vec.collectStatistics();
		vec.schema.computeDatasetStatistics();
		
		vec.debugStats();
		
		vec.schema.debugPrintRogueColumns();

		vec.schema.debugPrintDatasetStatistics();

	}

	
	@Test
	public void testSchemaLoad() {
		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("src/test/resources/", "src/test/resources/physionet_schema.txt" );
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
		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("/tmp/set-a/", "src/test/resources/physionet_schema.txt" );
		vec.loadSchema();

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
		
		

		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("/tmp/set-a/", "src/test/resources/physionet_schema.txt" );
		vec.loadSchema();
		
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
		
		INDArray input = Nd4j.zeros(new int[]{ 1, vec.schema.getTransformedVectorSize() + 1, 2 });
		
		vec.extractFileContentsAndVectorize( "src/test/resources/physionet_sample_data.txt", 0, vec.schema.getTransformedVectorSize() + 1, 2, input, null);
		
		PhysioNet_Vectorizer.debug3D_Nd4J_Input( input, 1, vec.schema.getTransformedVectorSize() + 1, 2 );

		
	}

	
	@Test
	public void testParseGeneralDescriptorValues() {
		
		String csvLine = "00:00,RecordID,135361";
		String[] columns = csvLine.split( "," );
		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("src/test/resources/", "src/test/resources/physionet_schema.txt");
		
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

	
}
