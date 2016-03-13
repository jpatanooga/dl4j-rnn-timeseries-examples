package org.deeplearning4j.jp.rnn.strata.physionet.utils;

import static org.junit.Assert.*;

import java.io.IOException;

import org.deeplearning4j.examples.rnn.strata.physionet.PhysioNet_Vectorizer;
import org.deeplearning4j.examples.rnn.strata.physionet.utils.PhysioNetDataUtils;
import org.deeplearning4j.examples.rnn.strata.physionet.utils.PhysioNetVectorizationDebugTool;
import org.junit.Test;

public class TestPhysioNetSubsetExtraction {

	@Test
	public void test() {
		//fail("Not yet implemented");
		
		//PhysioNetVectorizationDebugTool.extractBalancedSubsetOfPhysioNet( 40 );
		
		String physioNetBaseDirectory = "/tmp/set-a/";
		String physioSchemaFilePath = "src/test/resources/physionet_schema.txt";
		String physioLabelsFilePath = "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt";
		
		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer( physioNetBaseDirectory, physioSchemaFilePath, physioLabelsFilePath );
		vec.loadSchema();
		vec.loadLabels();
		//vec.setupFileInputList(false, 0);
		vec.setupBalancedSubset( 40 );
		vec.collectStatistics();
		vec.schema.debugPrintDatasetStatistics();
		
		
	}
	
	@Test
	public void testExtractFolds() throws IOException {
		
		
		PhysioNetDataUtils.extractNFoldSubset(true, "srcDirectory", "schemaPath", "labels_file_path", "/tmp/set-a-balanced-5/");
		
	}

}
