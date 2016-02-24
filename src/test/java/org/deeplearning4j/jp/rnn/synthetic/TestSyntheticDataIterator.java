package org.deeplearning4j.jp.rnn.synthetic;

import static org.junit.Assert.*;

import java.io.IOException;

import org.deeplearning4j.examples.rnn.synthetic.simple1.SyntheticDataIterator;
import org.junit.Test;

public class TestSyntheticDataIterator {

	@Test
	public void test() throws IOException {
		
		SyntheticDataIterator iter = new SyntheticDataIterator("src/test/resources/data/synthetic/simple/simple_ts_data", "", 40, 40);
		
		iter.next();
		
		
	}

}
