# Отчет по сравнению методов чанкирования

## Лучшие методы по метрикам:
- **faithfulness**: Sentence Window (0.9444)
- **answer_relevancy**: Markdown (0.6950)
- **context_precision**: Semantic (1.0000)
- **context_recall**: Markdown (1.0000)

## Общий вывод:
На основе усредненного показателя (Composite Score) лучшим методом является **Markdown** со счетом **0.8793**.

## Детальный разбор:
### Markdown
* Средний балл: 0.8793
* Сильные стороны: faithfulness, answer_relevancy, context_precision, context_recall
* Слабые стороны: 

### Sentence Window
* Средний балл: 0.8241
* Сильные стороны: faithfulness, context_precision, context_recall
* Слабые стороны: answer_relevancy

### Semantic
* Средний балл: 0.7591
* Сильные стороны: answer_relevancy, context_precision, context_recall
* Слабые стороны: faithfulness

### Recursive
* Средний балл: 0.6675
* Сильные стороны: context_recall
* Слабые стороны: faithfulness, answer_relevancy, context_precision

### Token
* Средний балл: 0.5544
* Сильные стороны: 
* Слабые стороны: faithfulness, answer_relevancy, context_precision, context_recall
