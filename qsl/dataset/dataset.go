package dataset

type Dataset interface {
	LoadQuerySamples([]int) error
	UnloadQuerySamples([]int) error
	GetSamples([]int) ([]interface{}, []interface{}, error)
	GetItemCount() int
}
