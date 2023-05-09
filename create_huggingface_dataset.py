    def _load_and_process():
        datasets = load_dataset('webis/Touche23-ValueEval')

        for split in datasets.keys():
            datasets[split] = datasets[split].map(
                self._preprocess_function,
                batched=True,
            )

        datasets.set_format(type='torch')
        return datasets

    def _load_dataset(self):

        if self.dataset_name_or_path == 'webis/Touche23-ValueEval':
            datasets = self.load_and_process()
        else:
            try:
                datasets = load_from_disk(self.dataset_name_or_path)
            except FileNotFoundError:
                print('Incorrect path. Dataset will be downloaded, processed, and saved to: ' +
                      self.dataset_name_or_path)
                datasets = self._load_and_process()
                datasets.save_to_disk(self.dataset_name_or_path)

        return datasets

    def _preprocess_function(self, examples, training=True):
        # Create input text by combining premise and hypothesis
        input_text = [
            f"premise: {premise}\n" f"hypothesis: {hypothesis}"
            for premise, hypothesis in zip(examples['premise'], examples['hypothesis'])
        ]

        # Tokenize input text
        model_inputs = self.tokenizer(
            examples['input_text'], truncation=True, max_length=self.max_source_length)

        # If we request the int labels for the classification task
        if self.classify:
            model_inputs["int_labels"] = examples["label"]
            return model_inputs

        # Tokenize first explanation and add as "labels" to model inputs
        targets = self.tokenizer(
            examples['explanation_1'], truncation=True, max_length=self.max_target_length)

        model_inputs["labels"] = targets["input_ids"]

        # Tokenize all explanations and assign to explanation_i
        if not training:
            for i in range(1, 4):
                key_explanation = f"explanation_{i}"
                targets = self.tokenizer(
                    examples[key_explanation], truncation=True, padding='max_length', max_length=self.max_target_length)
                model_inputs[key_explanation] = targets["input_ids"]
                # Note that these are zero padded and not -100 padded

        return model_inputs